"""Component tests R1-R8 plus query-API and integrity tests for the registry."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.registry.fingerprint import trial_fingerprint
from core.registry.models import (
    DecisionInput,
    ExperimentInput,
    LifecycleEventInput,
    ReproMetadata,
    ResultInput,
    StrategyInput,
    StrategyVersionInput,
)
from core.registry.store import (
    IllegalTransitionError,
    IncompleteReproMetadataError,
    InMemoryRegistryStore,
    RecordNotFoundError,
    RegistryStore,
    SecretInRecordError,
    validate_repro_metadata,
)
from tests.registry.conftest import (
    make_experiment_input,
    make_repro,
)

# --- R3: reproducibility round-trip + write precondition ---------------------


class TestReproRoundTripAndPrecondition:
    def test_experiment_round_trips_byte_identical(self, store: InMemoryRegistryStore) -> None:
        exp = make_experiment_input()
        exp_id = store.record_experiment(exp)
        fetched = store.get_experiment(exp_id)

        assert fetched.experiment_id == exp_id
        assert fetched.hypothesis == exp.hypothesis
        assert fetched.economic_rationale == exp.economic_rationale
        assert fetched.author == exp.author
        assert fetched.niche == exp.niche
        assert fetched.repro == exp.repro
        assert fetched.trial_fingerprint == trial_fingerprint(exp.repro)

    def test_incomplete_metadata_rejected_at_store(self, store: InMemoryRegistryStore) -> None:
        # Bypass model validation to simulate a smuggled-in incomplete bundle.
        broken = ReproMetadata.model_construct(**{**make_repro().__dict__, "code_sha": ""})
        exp = ExperimentInput.model_construct(
            **{**make_experiment_input().__dict__, "repro": broken}
        )
        with pytest.raises(IncompleteReproMetadataError, match="code_sha"):
            store.record_experiment(exp)

    def test_validate_repro_reports_missing_universe(self) -> None:
        bad = ReproMetadata.model_construct(
            **{**make_repro().__dict__, "data_snapshot": _SnapshotStub()}
        )
        with pytest.raises(IncompleteReproMetadataError, match="universe_spec"):
            validate_repro_metadata(bad)


class _SnapshotStub:
    as_of = None
    universe_spec: dict[str, object] = {}


# --- R4: trial fingerprint / dedup -------------------------------------------


class TestTrialDedup:
    def test_identical_trials_share_fingerprint_and_flagged_duplicate(
        self, store: InMemoryRegistryStore
    ) -> None:
        first = store.record_experiment(make_experiment_input())
        second = store.record_experiment(make_experiment_input())
        fp = store.get_experiment(first).trial_fingerprint
        assert store.get_experiment(second).trial_fingerprint == fp
        dups = store.duplicate_experiment_ids(fp)
        assert set(dups) == {first, second}

    def test_changed_params_new_fingerprint(self, store: InMemoryRegistryStore) -> None:
        a = store.record_experiment(make_experiment_input())
        b = store.record_experiment(
            make_experiment_input(repro=make_repro(params={"lookback": 99}))
        )
        fp_a = store.get_experiment(a).trial_fingerprint
        fp_b = store.get_experiment(b).trial_fingerprint
        assert fp_a != fp_b
        assert store.duplicate_experiment_ids(fp_b) == (b,)


# --- R1: immutability (store layer) ------------------------------------------


class TestImmutability:
    def test_store_exposes_no_update_or_delete(self) -> None:
        for forbidden in ("update", "delete", "set_status", "mutate", "edit"):
            assert not hasattr(RegistryStore, forbidden)

    def test_stored_record_is_frozen(self, store: InMemoryRegistryStore) -> None:
        from pydantic import ValidationError

        exp = store.get_experiment(store.record_experiment(make_experiment_input()))
        with pytest.raises(ValidationError):
            exp.hypothesis = "rewritten"  # type: ignore[misc]

    def test_correction_is_a_new_record(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        store.record_result(ResultInput(experiment_id=exp_id, status="failure"))
        store.record_result(ResultInput(experiment_id=exp_id, status="success"))
        results = store.results_for(exp_id)
        assert [r.status for r in results] == ["failure", "success"]


# --- R2: tamper evidence -----------------------------------------------------


class TestTamperEvidence:
    def test_clean_chain_verifies(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        store.record_result(ResultInput(experiment_id=exp_id, status="success"))
        store.record_decision(
            DecisionInput(
                experiment_id=exp_id,
                outcome="promote",
                deciding_component="gates.v1",
                component_version="1.0.0",
                rationale="passes all gates",
            )
        )
        assert store.verify_chain(exp_id).ok

    def test_edited_record_breaks_chain_at_exact_link(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        store.record_result(ResultInput(experiment_id=exp_id, status="success"))
        store.record_result(ResultInput(experiment_id=exp_id, status="failure"))

        # Simulate a retroactive edit to the seq-1 record without re-hashing.
        events = store._by_stream[exp_id]  # noqa: SLF001 — tamper simulation
        victim = events[1]
        tampered = dataclasses.replace(
            victim, payload={**victim.payload, "status": "success_FORGED"}
        )
        events[1] = tampered

        result = store.verify_chain(exp_id)
        assert result.ok is False
        assert result.broken_at_seq == 1
        assert result.broken_record_id == victim.record_id

    def test_empty_stream_verifies_ok(self, store: InMemoryRegistryStore) -> None:
        assert store.verify_chain("does_not_exist").ok


# --- R5: lifecycle event-sourcing --------------------------------------------


class TestLifecycleEventSourcing:
    def test_create_starts_in_candidate(self, store: InMemoryRegistryStore) -> None:
        sid = store.create_strategy(
            StrategyInput(name="drift-1", niche="elections", author="agent.x")
        )
        assert store.current_state(sid) == "candidate"

    def test_full_lifecycle_fold(self, store: InMemoryRegistryStore) -> None:
        sid = store.create_strategy(
            StrategyInput(name="drift-1", niche="elections", author="agent.x")
        )
        store.record_lifecycle_event(LifecycleEventInput(strategy_id=sid, event="promote"))
        store.record_lifecycle_event(LifecycleEventInput(strategy_id=sid, event="retire"))
        assert store.current_state(sid) == "retired"
        events = [e.event for e in store.lifecycle_events(sid)]
        assert events == ["create", "promote", "retire"]

    def test_retire_before_promote_rejected(self, store: InMemoryRegistryStore) -> None:
        sid = store.create_strategy(
            StrategyInput(name="drift-1", niche="elections", author="agent.x")
        )
        with pytest.raises(IllegalTransitionError):
            store.record_lifecycle_event(LifecycleEventInput(strategy_id=sid, event="retire"))
        # The rejected transition was not persisted.
        assert store.current_state(sid) == "candidate"

    def test_strategies_by_state(self, store: InMemoryRegistryStore) -> None:
        promoted = store.create_strategy(StrategyInput(name="s-prom", niche="n", author="a"))
        store.record_lifecycle_event(LifecycleEventInput(strategy_id=promoted, event="promote"))
        candidate = store.create_strategy(StrategyInput(name="s-cand", niche="n", author="a"))
        assert [s.strategy_id for s in store.strategies_by_state("promoted")] == [promoted]
        assert [s.strategy_id for s in store.strategies_by_state("candidate")] == [candidate]


# --- R6: failures first-class ------------------------------------------------


class TestFailuresFirstClass:
    def _decide(self, store: RegistryStore, exp_id: str, outcome: str) -> None:
        store.record_decision(
            DecisionInput(
                experiment_id=exp_id,
                outcome=outcome,  # type: ignore[arg-type]
                deciding_component="gates.v1",
                component_version="1.0.0",
                rationale=f"decided {outcome}",
            )
        )

    def test_rejected_and_abandoned_queryable_by_outcome_and_niche(
        self, store: InMemoryRegistryStore
    ) -> None:
        promoted = store.record_experiment(make_experiment_input(niche="onc"))
        rejected = store.record_experiment(make_experiment_input(niche="onc"))
        abandoned = store.record_experiment(make_experiment_input(niche="onc"))
        self._decide(store, promoted, "promote")
        self._decide(store, rejected, "reject")
        self._decide(store, abandoned, "abandon")

        assert [e.experiment_id for e in store.experiments_by_outcome("reject")] == [rejected]
        assert [e.experiment_id for e in store.experiments_by_outcome("abandon")] == [abandoned]
        by_niche = {e.experiment_id for e in store.experiments_by_niche("onc")}
        assert by_niche == {promoted, rejected, abandoned}

    def test_all_experiments_returns_full_set(self, store: InMemoryRegistryStore) -> None:
        first = store.record_experiment(make_experiment_input(niche="onc"))
        second = store.record_experiment(make_experiment_input(niche="cardio"))
        assert {e.experiment_id for e in store.all_experiments()} == {first, second}

    def test_latest_decision_wins(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        self._decide(store, exp_id, "reject")
        self._decide(store, exp_id, "promote")  # a later correction
        assert [e.experiment_id for e in store.experiments_by_outcome("promote")] == [exp_id]
        assert store.experiments_by_outcome("reject") == ()

    def test_failure_result_is_retrievable(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        store.record_result(
            ResultInput(experiment_id=exp_id, status="failure", metrics={"sharpe": -0.3})
        )
        results = store.results_for(exp_id)
        assert results[0].status == "failure"
        assert results[0].metrics["sharpe"] == -0.3


# --- R7: lineage -------------------------------------------------------------


class TestLineage:
    def test_parent_child_traversal(self, store: InMemoryRegistryStore) -> None:
        a = store.record_experiment(make_experiment_input())
        b = store.record_experiment(make_experiment_input(parent_experiment_id=a))
        c = store.record_experiment(make_experiment_input(parent_experiment_id=b))

        assert [e.experiment_id for e in store.experiment_lineage(c)] == [a, b, c]
        assert [e.experiment_id for e in store.experiment_children(a)] == [b]
        assert store.experiment_children(c) == ()

    def test_unknown_parent_rejected(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError):
            store.record_experiment(make_experiment_input(parent_experiment_id="exp_missing"))

    def test_strategy_version_resolves_full_ancestry(self, store: InMemoryRegistryStore) -> None:
        a = store.record_experiment(make_experiment_input())
        b = store.record_experiment(make_experiment_input(parent_experiment_id=a))
        sid = store.create_strategy(StrategyInput(name="s", niche="n", author="agent.x"))
        store.record_strategy_version(
            StrategyVersionInput(
                strategy_id=sid,
                version=1,
                origin_experiment_id=b,
                spec_hash="spec-hash-001",
            )
        )
        ancestry = store.strategy_version_ancestry(sid, 1)
        assert [e.experiment_id for e in ancestry] == [a, b]

    def test_strategy_version_requires_existing_origin(self, store: InMemoryRegistryStore) -> None:
        sid = store.create_strategy(StrategyInput(name="s", niche="n", author="agent.x"))
        with pytest.raises(RecordNotFoundError):
            store.record_strategy_version(
                StrategyVersionInput(
                    strategy_id=sid,
                    version=1,
                    origin_experiment_id="exp_missing",
                    spec_hash="x",
                )
            )


# --- R8: concurrency ---------------------------------------------------------


class TestConcurrency:
    def test_concurrent_appends_to_different_streams(self) -> None:
        # Default wall-clock backend (thread-safe); each experiment is its own stream.
        store = InMemoryRegistryStore()

        def worker(i: int) -> str:
            return store.record_experiment(make_experiment_input(niche=f"n{i}"))

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(worker, range(50)))

        assert len(set(ids)) == 50
        for exp_id in ids:
            assert store.verify_chain(exp_id).ok

    def test_concurrent_appends_to_same_stream_keep_valid_chain(self) -> None:
        store = InMemoryRegistryStore()
        exp_id = store.record_experiment(make_experiment_input())

        def add_result(i: int) -> str:
            return store.record_result(
                ResultInput(experiment_id=exp_id, status="success", metrics={"i": i})
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(add_result, range(40)))

        chain = store.verify_chain(exp_id)
        assert chain.ok
        events = store._stream_events(exp_id)  # noqa: SLF001
        assert [e.seq for e in events] == list(range(41))  # 1 experiment + 40 results


# --- secrets + misc integrity ------------------------------------------------


class TestSecretsAndMisc:
    def test_secret_in_params_rejected(self, store: InMemoryRegistryStore) -> None:
        exp = make_experiment_input(
            repro=make_repro(params={"api_key": "sk-secret", "lookback": 20})
        )
        with pytest.raises(SecretInRecordError):
            store.record_experiment(exp)

    def test_secret_in_decision_evidence_rejected(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        with pytest.raises(SecretInRecordError):
            store.record_decision(
                DecisionInput(
                    experiment_id=exp_id,
                    outcome="promote",
                    deciding_component="gates.v1",
                    component_version="1.0.0",
                    rationale="ok",
                    evidence={"aws_secret_access_key": "AKIA..."},
                )
            )

    def test_result_for_unknown_experiment_rejected(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError):
            store.record_result(ResultInput(experiment_id="exp_missing", status="success"))

    def test_experiments_by_author(self, store: InMemoryRegistryStore) -> None:
        mine = store.record_experiment(make_experiment_input(author="agent.alpha"))
        store.record_experiment(make_experiment_input(author="agent.beta"))
        assert [e.experiment_id for e in store.experiments_by_author("agent.alpha")] == [mine]

    def test_get_unknown_strategy_rejected(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError):
            store.get_strategy("strat_missing")

    def test_decisions_for_returns_all_in_order(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        store.record_decision(
            DecisionInput(
                experiment_id=exp_id,
                outcome="reject",
                deciding_component="gates.v1",
                component_version="1.0.0",
                rationale="first",
            )
        )
        store.record_decision(
            DecisionInput(
                experiment_id=exp_id,
                outcome="promote",
                deciding_component="gates.v1",
                component_version="1.0.1",
                rationale="second",
            )
        )
        decisions = store.decisions_for(exp_id)
        assert [d.outcome for d in decisions] == ["reject", "promote"]

    def test_strategy_version_ancestry_unknown_version_rejected(
        self, store: InMemoryRegistryStore
    ) -> None:
        sid = store.create_strategy(StrategyInput(name="s", niche="n", author="agent.x"))
        with pytest.raises(RecordNotFoundError):
            store.strategy_version_ancestry(sid, 99)

    def test_secret_nested_in_list_rejected(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        with pytest.raises(SecretInRecordError):
            store.record_result(
                ResultInput(
                    experiment_id=exp_id,
                    status="success",
                    artifacts={"logs": [{"api_key": "leak"}]},
                )
            )


class TestReproPreconditionBranches:
    def test_missing_data_snapshot_reported(self) -> None:
        bad = ReproMetadata.model_construct(**{**make_repro().__dict__, "data_snapshot": None})
        with pytest.raises(IncompleteReproMetadataError, match="data_snapshot"):
            validate_repro_metadata(bad)

    def test_missing_params_and_seeds_reported(self) -> None:
        bad = ReproMetadata.model_construct(
            **{**make_repro().__dict__, "params": None, "seeds": None}
        )
        with pytest.raises(IncompleteReproMetadataError, match="params"):
            validate_repro_metadata(bad)

    def test_missing_spec_hash_reported(self) -> None:
        bad = ReproMetadata.model_construct(**{**make_repro().__dict__, "spec_hash": ""})
        with pytest.raises(IncompleteReproMetadataError, match="spec_hash"):
            validate_repro_metadata(bad)
