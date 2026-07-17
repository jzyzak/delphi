"""Unit tests for the calibrate + uncertainty stage (C4.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from core.forecast.supervisor import Confidence, DisagreementKind, ReconciledForecast
from forecaster.stages.calibrate import IdentityRecalibrator, calibrate_reconciled

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _reconciled(probability: float, uncertainty: float = 0.05) -> ReconciledForecast:
    return ReconciledForecast(
        probability=probability,
        uncertainty=uncertainty,
        aggregate_probability=probability,
        confidence=Confidence.LOW,
        applied=False,
        knowledge_time=AS_OF,
        disagreement=DisagreementKind.NONE,
    )


def test_identity_recalibrator_then_extremizes() -> None:
    calibrated, uncertainty = calibrate_reconciled(_reconciled(0.7))
    # Extremization pushes a >0.5 probability further toward 1.
    assert calibrated.calibrated_probability > 0.7
    assert calibrated.raw_probability == 0.7
    assert uncertainty.combined >= 0.0
    assert calibrated.provenance["recalibrator"]["fitted"] is False


def test_midpoint_is_fixed() -> None:
    calibrated, _ = calibrate_reconciled(_reconciled(0.5))
    assert calibrated.calibrated_probability == pytest.approx(0.5)


def test_custom_recalibrator_is_applied_before_extremization() -> None:
    class ToHalf:
        def apply(self, probability: float) -> float:
            return 0.5

        @property
        def provenance(self) -> dict[str, Any]:
            return {"recalibrator": "to_half", "fitted": True}

    calibrated, _ = calibrate_reconciled(_reconciled(0.9), recalibrator=ToHalf())
    # Recalibrated to 0.5, so extremization keeps it at 0.5.
    assert calibrated.calibrated_probability == pytest.approx(0.5)
    assert calibrated.provenance["recalibrated_probability"] == 0.5


def test_identity_recalibrator_apply() -> None:
    assert IdentityRecalibrator().apply(0.42) == 0.42


def test_high_spread_flags_boundary() -> None:
    calibrated, _ = calibrate_reconciled(_reconciled(0.7, uncertainty=0.3))
    assert calibrated.near_decision_boundary is True


def test_floor_clamps_the_final_probability() -> None:
    calibrated, _ = calibrate_reconciled(_reconciled(0.01), floor=0.05)
    assert calibrated.calibrated_probability == pytest.approx(0.05)
    assert calibrated.provenance["floor"] == 0.05


def test_floor_clamps_the_high_tail_symmetrically() -> None:
    calibrated, _ = calibrate_reconciled(_reconciled(0.99), floor=0.05)
    assert calibrated.calibrated_probability == pytest.approx(0.95)


def test_no_floor_leaves_extremized_tail() -> None:
    calibrated, _ = calibrate_reconciled(_reconciled(0.01))
    assert calibrated.calibrated_probability < 0.01
    assert calibrated.provenance["floor"] is None


def test_interior_probability_is_not_floored() -> None:
    with_floor, _ = calibrate_reconciled(_reconciled(0.7), floor=0.05)
    without, _ = calibrate_reconciled(_reconciled(0.7))
    assert with_floor.calibrated_probability == pytest.approx(without.calibrated_probability)


def test_invalid_floor_raises() -> None:
    with pytest.raises(ValueError, match="floor"):
        calibrate_reconciled(_reconciled(0.7), floor=0.5)


def test_fitted_alpha_overrides_default() -> None:
    neutral, _ = calibrate_reconciled(_reconciled(0.7), alpha=1.0)
    assert neutral.calibrated_probability == pytest.approx(0.7)
    assert neutral.provenance["alpha"] == 1.0
