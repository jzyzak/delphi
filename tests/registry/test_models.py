"""Unit tests for registry models: repro contract + lifecycle state machine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from core.registry.models import (
    DataSnapshot,
    EnvFingerprint,
    IllegalTransitionError,
    fold_lifecycle,
    next_state,
)
from tests.registry.conftest import AS_OF, make_repro


class TestReproContractValidation:
    def test_complete_metadata_constructs(self) -> None:
        assert make_repro().code_sha == "a1b2c3d4"

    def test_blank_code_sha_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_repro(code_sha="   ")

    def test_blank_spec_hash_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_repro(spec_hash="")

    def test_empty_universe_spec_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataSnapshot(as_of=AS_OF, universe_spec={})

    def test_blank_python_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EnvFingerprint(python_version="")

    def test_naive_as_of_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataSnapshot(as_of=datetime(2024, 6, 1), universe_spec={"classification": "elections"})

    def test_as_of_normalized_to_utc(self) -> None:
        from datetime import timedelta, timezone

        eastern = timezone(timedelta(hours=-5))
        snap = DataSnapshot(
            as_of=datetime(2024, 6, 1, 12, tzinfo=eastern),
            universe_spec={"classification": "elections"},
        )
        assert snap.as_of.tzinfo == UTC
        assert snap.as_of.hour == 17

    def test_repro_is_frozen(self) -> None:
        repro = make_repro()
        with pytest.raises(ValidationError):
            repro.code_sha = "mutated"  # type: ignore[misc]


class TestLifecycleStateMachine:
    def test_create_yields_candidate(self) -> None:
        assert next_state(None, "create") == "candidate"

    def test_promote_from_candidate(self) -> None:
        assert next_state("candidate", "promote") == "promoted"

    def test_retire_from_promoted(self) -> None:
        assert next_state("promoted", "retire") == "retired"

    def test_retire_before_promote_is_illegal(self) -> None:
        with pytest.raises(IllegalTransitionError):
            next_state("candidate", "retire")

    def test_promote_twice_is_illegal(self) -> None:
        with pytest.raises(IllegalTransitionError):
            next_state("promoted", "promote")

    def test_no_transition_out_of_retired(self) -> None:
        with pytest.raises(IllegalTransitionError):
            next_state("retired", "promote")

    def test_create_after_creation_is_illegal(self) -> None:
        with pytest.raises(IllegalTransitionError):
            next_state("candidate", "create")

    def test_fold_empty_is_none(self) -> None:
        assert fold_lifecycle([]) is None

    def test_fold_full_lifecycle(self) -> None:
        assert fold_lifecycle(["create", "promote", "retire"]) == "retired"

    def test_fold_partial(self) -> None:
        assert fold_lifecycle(["create", "promote"]) == "promoted"

    def test_fold_rejects_illegal_sequence(self) -> None:
        with pytest.raises(IllegalTransitionError):
            fold_lifecycle(["create", "retire"])


class TestInputModelValidation:
    def test_experiment_blank_hypothesis_rejected(self) -> None:
        from tests.registry.conftest import make_experiment_input

        with pytest.raises(ValidationError):
            make_experiment_input(hypothesis="  ")

    def test_decision_blank_component_rejected(self) -> None:
        from core.registry.models import DecisionInput

        with pytest.raises(ValidationError):
            DecisionInput(
                experiment_id="exp_1",
                outcome="promote",
                deciding_component=" ",
                component_version="1.0.0",
                rationale="ok",
            )

    def test_strategy_blank_name_rejected(self) -> None:
        from core.registry.models import StrategyInput

        with pytest.raises(ValidationError):
            StrategyInput(name="", niche="n", author="a")

    def test_strategy_version_must_be_positive(self) -> None:
        from core.registry.models import StrategyVersionInput

        with pytest.raises(ValidationError):
            StrategyVersionInput(
                strategy_id="s", version=0, origin_experiment_id="e", spec_hash="x"
            )
