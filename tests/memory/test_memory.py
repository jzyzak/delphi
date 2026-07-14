"""Component tests M1-M7 plus §8 unit tests for agent memory."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.memory.embedder import DeterministicEmbedder
from core.memory.index import (
    DimensionMismatchError,
    InMemoryVectorIndex,
    PostgresVectorIndex,
    assemble_document,
    render_spec_description,
)
from core.memory.recall import MemoryRecall
from core.registry.fingerprint import trial_fingerprint
from core.registry.models import DecisionInput
from core.registry.store import InMemoryRegistryStore, RecordNotFoundError
from tests.memory.conftest import (
    ClusteredFixtureEmbedder,
    record_experiment_with_outcome,
)
from tests.registry.conftest import make_experiment_input, make_repro

pytestmark_postgres = pytest.mark.postgres


# --- M1: semantic recall -----------------------------------------------------


class TestSemanticRecall:
    def test_relevant_experiments_rank_above_irrelevant(
        self,
        recall: MemoryRecall,
        memory_index: InMemoryVectorIndex,
        store: InMemoryRegistryStore,
    ) -> None:
        elections_id = record_experiment_with_outcome(
            store,
            hypothesis="Polling drift in close election races.",
            niche="us_elections",
            outcome="reject",
            rationale="No edge after costs.",
        )
        weather_id = record_experiment_with_outcome(
            store,
            hypothesis="Hurricane landfall count momentum.",
            niche="atlantic_weather",
            outcome="promote",
            rationale="Stable edge across folds.",
        )
        memory_index.index(store.get_experiment(elections_id))
        memory_index.index(store.get_experiment(weather_id))

        hits = recall.recall(
            query="election polling drift hypothesis",
            k=2,
        )

        assert len(hits) == 2
        assert hits[0].experiment_id == elections_id
        assert hits[0].score > hits[1].score


# --- M2: failures first-class ------------------------------------------------


class TestFailuresFirstClass:
    def test_rejected_and_abandoned_retrievable_by_niche_and_outcome(
        self,
        recall: MemoryRecall,
        memory_index: InMemoryVectorIndex,
        store: InMemoryRegistryStore,
    ) -> None:
        rejected_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling fade after primaries.",
            niche="us_elections",
            outcome="reject",
            rationale="Turnover too high.",
        )
        abandoned_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling pre-debate movement.",
            niche="us_elections",
            outcome="abandon",
            rationale="Capacity insufficient.",
        )
        promoted_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling post-debate drift.",
            niche="us_elections",
            outcome="promote",
            rationale="Positive deflated Sharpe.",
        )
        for exp_id in (rejected_id, abandoned_id, promoted_id):
            memory_index.index(store.get_experiment(exp_id))

        rejected_hits = recall.recall(
            query="election polling",
            niche="us_elections",
            outcome="rejected",
            k=5,
        )
        abandoned_hits = recall.recall(
            query="election polling",
            niche="us_elections",
            outcome="abandoned",
            k=5,
        )

        assert {hit.experiment_id for hit in rejected_hits} == {rejected_id}
        assert {hit.experiment_id for hit in abandoned_hits} == {abandoned_id}


# --- M3/M7: rebuildable + incremental ----------------------------------------


class TestRebuildableIndex:
    def test_rebuild_matches_incremental_recall(
        self,
        store: InMemoryRegistryStore,
        embedder: ClusteredFixtureEmbedder,
    ) -> None:
        exp_ids = [
            record_experiment_with_outcome(
                store,
                hypothesis="Election polling drift variant A.",
                niche="us_elections",
                outcome="reject",
                rationale="Lesson A.",
            ),
            record_experiment_with_outcome(
                store,
                hypothesis="Atlantic hurricane weather momentum.",
                niche="atlantic_weather",
                outcome="promote",
                rationale="Lesson B.",
            ),
        ]

        incremental = InMemoryVectorIndex(store, embedder)
        for exp_id in exp_ids:
            incremental.index(store.get_experiment(exp_id))
        incremental_recall = MemoryRecall(embedder, incremental, store)
        incremental_hits = incremental_recall.recall(query="election polling drift", k=5)

        rebuilt = InMemoryVectorIndex(store, embedder)
        rebuilt.rebuild_from_registry()
        rebuilt_recall = MemoryRecall(embedder, rebuilt, store)
        rebuilt_hits = rebuilt_recall.recall(query="election polling drift", k=5)

        assert [hit.experiment_id for hit in rebuilt_hits] == [
            hit.experiment_id for hit in incremental_hits
        ]
        assert [hit.score for hit in rebuilt_hits] == [hit.score for hit in incremental_hits]

    def test_new_experiment_recallable_after_incremental_index(
        self,
        store: InMemoryRegistryStore,
        embedder: ClusteredFixtureEmbedder,
    ) -> None:
        index = InMemoryVectorIndex(store, embedder)
        recall_api = MemoryRecall(embedder, index, store)

        exp_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling event study.",
            niche="us_elections",
            outcome="reject",
            rationale="No signal.",
        )
        index.index(store.get_experiment(exp_id))

        hits = recall_api.recall(query="election polling", k=1)
        assert hits[0].experiment_id == exp_id

        index.rebuild_from_registry()
        rebuilt_hits = recall_api.recall(query="election polling", k=1)
        assert rebuilt_hits[0].experiment_id == exp_id


# --- M4: near-duplicate advisory ---------------------------------------------


class TestNearDuplicateAdvisory:
    def test_near_duplicate_flags_semantic_match_not_fingerprint(
        self,
        recall: MemoryRecall,
        memory_index: InMemoryVectorIndex,
        store: InMemoryRegistryStore,
    ) -> None:
        repro = make_repro(spec_hash="spec-hash-A", params={"lookback": 20})
        exp_id = store.record_experiment(
            make_experiment_input(
                hypothesis="Election polling drift after primaries.",
                niche="us_elections",
                repro=repro,
            )
        )
        store.record_decision(
            DecisionInput(
                experiment_id=exp_id,
                outcome="reject",
                deciding_component="harness.gates",
                component_version="1.0.0",
                rationale="Failed gate.",
            )
        )
        memory_index.index(store.get_experiment(exp_id))

        different_params = make_repro(spec_hash="spec-hash-B", params={"lookback": 30})
        candidate_spec = render_spec_description(different_params)
        assert trial_fingerprint(repro) != trial_fingerprint(different_params)

        flagged = recall.near_duplicates(
            spec_description=candidate_spec + " election polling drift signal",
            threshold=0.5,
        )

        assert flagged
        assert flagged[0].experiment_id == exp_id
        assert flagged[0].trial_fingerprint == store.get_experiment(exp_id).trial_fingerprint


# --- M5: determinism ---------------------------------------------------------


class TestDeterminism:
    def test_fixed_embedder_yields_identical_recall(
        self,
        store: InMemoryRegistryStore,
        embedder: ClusteredFixtureEmbedder,
    ) -> None:
        exp_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling drift.",
            niche="us_elections",
            outcome="reject",
            rationale="No edge.",
        )
        index_a = InMemoryVectorIndex(store, embedder)
        index_b = InMemoryVectorIndex(store, embedder)
        index_a.index(store.get_experiment(exp_id))
        index_b.index(store.get_experiment(exp_id))

        recall_a = MemoryRecall(embedder, index_a, store)
        recall_b = MemoryRecall(embedder, index_b, store)

        hits_a = recall_a.recall(query="election polling", k=3)
        hits_b = recall_b.recall(query="election polling", k=3)

        assert hits_a == hits_b


# --- M6: no secrets / mockable -----------------------------------------------


class TestNoSecretsMockable:
    def test_embedded_text_has_no_secret_markers(
        self,
        store: InMemoryRegistryStore,
    ) -> None:
        exp_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling drift.",
            niche="us_elections",
            outcome="reject",
            rationale="Underperformed after borrow costs.",
        )
        doc = assemble_document(store, store.get_experiment(exp_id))
        lowered = doc.embedded_text.lower()
        for marker in ("password", "secret", "api_key", "token", "credential"):
            assert marker not in lowered

    def test_embedder_is_mockable_protocol(self) -> None:
        mock = MagicMock()
        mock.dim = 4
        mock.embed.return_value = [[1.0, 0.0, 0.0, 0.0]]
        index = InMemoryVectorIndex(InMemoryRegistryStore(), mock)
        hits = index.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert hits == []


# --- §8: unit tests ------------------------------------------------------------


class TestEmbedder:
    def test_deterministic_embedder_same_text_same_vector(self) -> None:
        embedder = DeterministicEmbedder(dim=16)
        first = embedder.embed(["election polling drift"])
        second = embedder.embed(["election polling drift"])
        assert first == second

    def test_deterministic_embedder_rejects_small_dim(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            DeterministicEmbedder(dim=4)

    def test_deterministic_embedder_empty_batch(self) -> None:
        embedder = DeterministicEmbedder()
        assert embedder.embed([]) == []


class TestAssembleDocument:
    def test_pending_outcome_without_decision(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        doc = assemble_document(store, store.get_experiment(exp_id))
        assert doc.outcome == "pending"
        assert "Hypothesis:" in doc.embedded_text

    def test_unknown_experiment_raises(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        exp = store.get_experiment(exp_id)
        fake = exp.model_copy(update={"experiment_id": "exp_missing"})
        with pytest.raises(RecordNotFoundError):
            assemble_document(store, fake)


class TestRecallValidation:
    def test_empty_query_rejected(self, recall: MemoryRecall) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            recall.recall(query="   ")

    def test_invalid_threshold_rejected(self, recall: MemoryRecall) -> None:
        with pytest.raises(ValueError, match="threshold"):
            recall.near_duplicates(spec_description="dsl spec", threshold=1.5)

    def test_negative_k_rejected(self, recall: MemoryRecall) -> None:
        with pytest.raises(ValueError, match="k must"):
            recall.recall(query="election", k=-1)


class TestRegistryEnumeration:
    def test_all_experiments_returns_every_record(self, store: InMemoryRegistryStore) -> None:
        first = store.record_experiment(make_experiment_input(hypothesis="First."))
        second = store.record_experiment(make_experiment_input(hypothesis="Second."))
        assert {exp.experiment_id for exp in store.all_experiments()} == {first, second}


class TestLessonsRetrieval:
    def test_lessons_returns_distilled_strings(
        self,
        recall: MemoryRecall,
        memory_index: InMemoryVectorIndex,
        store: InMemoryRegistryStore,
    ) -> None:
        exp_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling drift.",
            niche="us_elections",
            outcome="reject",
            rationale="Borrow costs erased edge.",
        )
        memory_index.index(store.get_experiment(exp_id))

        lessons = recall.lessons(query="election polling", k=1)
        assert len(lessons) == 1
        assert lessons[0].startswith("Borrow costs erased edge.")
        assert "evidence=" in lessons[0]
        assert "metrics=" in lessons[0]


class TestDimensionMismatch:
    def test_search_rejects_mismatched_vector(
        self,
        store: InMemoryRegistryStore,
        embedder: ClusteredFixtureEmbedder,
    ) -> None:
        exp_id = record_experiment_with_outcome(
            store,
            hypothesis="Election polling drift.",
            niche="us_elections",
            outcome="reject",
            rationale="No edge.",
        )
        index = InMemoryVectorIndex(store, embedder)
        index.index(store.get_experiment(exp_id))
        with pytest.raises(DimensionMismatchError):
            index.search([1.0, 0.0], k=1)


class TestPostgresIntegration:
    @pytest.mark.postgres
    def test_postgres_round_trip_recall(
        self,
        postgres_memory_stack: tuple[Any, Any, MemoryRecall],
    ) -> None:
        registry, index, recall_api = postgres_memory_stack
        exp_id = record_experiment_with_outcome(
            registry,
            hypothesis="Election polling drift in Postgres.",
            niche="us_elections",
            outcome="reject",
            rationale="Failed validation.",
        )
        index.index(registry.get_experiment(exp_id))
        hits = recall_api.recall(query="election polling drift", niche="us_elections", k=1)
        assert hits[0].experiment_id == exp_id

    @pytest.mark.postgres
    def test_postgres_rebuild_matches_incremental(
        self,
        postgres_memory_stack: tuple[Any, Any, MemoryRecall],
    ) -> None:
        registry, index, recall_api = postgres_memory_stack
        exp_id = record_experiment_with_outcome(
            registry,
            hypothesis="Election polling rebuild test.",
            niche="us_elections",
            outcome="abandon",
            rationale="Capacity too low.",
        )
        index.index(registry.get_experiment(exp_id))
        incremental = recall_api.recall(query="election polling rebuild", k=1)

        index.rebuild_from_registry()
        rebuilt = recall_api.recall(query="election polling rebuild", k=1)

        assert incremental == rebuilt


class TestPostgresMocked:
    def test_apply_migrations_executes_sql_files(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        conn = MagicMock()
        store = InMemoryRegistryStore()
        embedder = DeterministicEmbedder()
        monkeypatch.setattr("core.memory.index.register_vector", lambda _conn: None)
        index = PostgresVectorIndex(conn, store, embedder)
        migration = tmp_path / "0001_test.sql"
        migration.write_text("SELECT 1;")
        monkeypatch.setattr("core.memory.index._MIGRATIONS_DIR", tmp_path)
        index.apply_migrations()
        conn.commit.assert_called_once()
