"""Tests for calibration split assignment + recalibrator/extremization fit (C6.5)."""

from __future__ import annotations

import pytest

from evaluation.calibration_split import (
    MIN_ISOTONIC_SAMPLES,
    CalibrationArtifact,
    IsotonicRecalibrator,
    assign_calibration_split,
    fit_calibration_artifact,
    fit_extremization_coefficient,
    fit_isotonic,
)
from evaluation.scoring import ScoredRecord
from forecaster.stages.calibrate import Recalibrator


def _records(pairs: list[tuple[float, float]]) -> list[ScoredRecord]:
    return [
        ScoredRecord(question_id=f"q{i}", domain="d", probability=p, outcome=o)
        for i, (p, o) in enumerate(pairs)
    ]


class TestSplitAssignment:
    def test_disjoint_from_holdout_and_live(self) -> None:
        ids = [f"q{i}" for i in range(20)]
        holdout = {"q0", "q1", "q2"}
        live = {"q3", "q4"}
        split = assign_calibration_split(ids, holdout_ids=holdout, live_ids=live, seed=0)
        assert not (split & holdout)
        assert not (split & live)
        assert split == frozenset(ids) - holdout - live

    def test_fraction_subselects(self) -> None:
        ids = [f"q{i}" for i in range(10)]
        split = assign_calibration_split(ids, fraction=0.5, seed=0)
        assert len(split) == 5

    def test_deterministic(self) -> None:
        ids = [f"q{i}" for i in range(10)]
        assert assign_calibration_split(ids, fraction=0.4, seed=3) == assign_calibration_split(
            ids, fraction=0.4, seed=3
        )

    def test_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            assign_calibration_split(["q0"], fraction=1.5)


class TestIsotonic:
    def test_monotone_fit(self) -> None:
        recal = fit_isotonic([0.1, 0.3, 0.6, 0.9], [0.0, 0.0, 1.0, 1.0])
        assert recal.apply(0.1) <= recal.apply(0.6) <= recal.apply(0.9)

    def test_pools_violators(self) -> None:
        # Outcomes decrease then increase; PAV must produce a monotone fit.
        recal = fit_isotonic([0.2, 0.4, 0.6, 0.8], [1.0, 0.0, 0.0, 1.0])
        vals = [recal.apply(x) for x in (0.2, 0.4, 0.6, 0.8)]
        assert vals == sorted(vals)

    def test_apply_clamped(self) -> None:
        recal = fit_isotonic([0.0, 1.0], [0.0, 1.0])
        assert 0.0 < recal.apply(0.0) < 1.0
        assert 0.0 < recal.apply(1.0) < 1.0

    def test_provenance_and_roundtrip(self) -> None:
        recal = fit_isotonic([0.1, 0.9], [0.0, 1.0])
        assert recal.provenance["fitted"] is True
        restored = IsotonicRecalibrator.from_dict(recal.to_dict())
        assert restored.apply(0.5) == pytest.approx(recal.apply(0.5))

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            fit_isotonic([0.1], [0.0, 1.0])
        with pytest.raises(ValueError, match="empty"):
            fit_isotonic([], [])

    def test_satisfies_recalibrator_protocol(self) -> None:
        recal = fit_isotonic([0.1, 0.9], [0.0, 1.0])
        assert isinstance(recal, Recalibrator)


class TestExtremization:
    def test_picks_from_grid(self) -> None:
        # Confident-correct forecasts favor extremization > 1.
        probs = [0.7, 0.7, 0.3, 0.3]
        outcomes = [1.0, 1.0, 0.0, 0.0]
        alpha = fit_extremization_coefficient(probs, outcomes, grid=(1.0, 2.0))
        assert alpha == 2.0

    def test_keeps_best_when_later_grid_value_is_worse(self) -> None:
        # Best alpha (2.0) comes first; the worse 1.0 must not overwrite it.
        probs = [0.7, 0.7, 0.3, 0.3]
        outcomes = [1.0, 1.0, 0.0, 0.0]
        assert fit_extremization_coefficient(probs, outcomes, grid=(2.0, 1.0)) == 2.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="empty set"):
            fit_extremization_coefficient([], [])
        with pytest.raises(ValueError, match="grid"):
            fit_extremization_coefficient([0.5], [1.0], grid=())

    def test_default_grid_reaches_below_old_minimum(self) -> None:
        # Overconfident-wrong forecasts favor de-extremization; the widened
        # default grid now reaches 0.25 (the old grid bottomed out at 0.5).
        probs = [0.9, 0.1, 0.8, 0.2]
        outcomes = [0.0, 1.0, 0.0, 1.0]
        assert fit_extremization_coefficient(probs, outcomes) == 0.25

    def test_default_grid_reaches_above_old_maximum(self) -> None:
        # Mostly-correct confident forecasts with one miss have an interior
        # log-loss minimum at 2.5 (the old grid capped at 2.0).
        probs = [0.7] * 9
        outcomes = [1.0] * 8 + [0.0]
        assert fit_extremization_coefficient(probs, outcomes) == 2.5


class TestArtifact:
    def test_fit_and_roundtrip(self) -> None:
        records = _records([(0.1, 0.0), (0.4, 0.0), (0.6, 1.0), (0.9, 1.0)])
        artifact = fit_calibration_artifact(records)
        restored = CalibrationArtifact.from_dict(artifact.to_dict())
        assert restored.alpha == pytest.approx(artifact.alpha)
        assert restored.recalibrator.apply(0.5) == pytest.approx(artifact.recalibrator.apply(0.5))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty split"):
            fit_calibration_artifact([])


class TestMinSampleGuard:
    def test_small_split_uses_identity_but_still_fits_alpha(self) -> None:
        # 5 samples < MIN_ISOTONIC_SAMPLES: isotonic would overfit, so the
        # recalibrator is the identity map — but alpha is still grid-fitted
        # (confident-correct records push it to the grid maximum, 3.0).
        records = _records([(0.8, 1.0), (0.7, 1.0), (0.3, 0.0), (0.2, 0.0), (0.9, 1.0)])
        artifact = fit_calibration_artifact(records)
        for p in (0.1, 0.25, 0.5, 0.73, 0.9):
            assert artifact.recalibrator.apply(p) == pytest.approx(p)
        assert artifact.recalibrator.provenance == {
            "recalibrator": "identity",
            "fitted": False,
            "n": 0,
        }
        assert artifact.alpha == 3.0

    def test_exactly_min_samples_fits_isotonic(self) -> None:
        # At the MIN_ISOTONIC_SAMPLES boundary (20) the isotonic path is taken.
        records = _records([(0.05 * (i + 1), 1.0 if i >= 10 else 0.0) for i in range(20)])
        assert len(records) == MIN_ISOTONIC_SAMPLES
        artifact = fit_calibration_artifact(records)
        assert artifact.recalibrator.n == 20
        assert artifact.recalibrator.provenance["recalibrator"] == "isotonic_pav"
        assert artifact.recalibrator.provenance["fitted"] is True

    def test_deterministic(self) -> None:
        records = _records([(0.8, 1.0), (0.7, 1.0), (0.3, 0.0), (0.2, 0.0), (0.9, 1.0)])
        assert fit_calibration_artifact(records) == fit_calibration_artifact(records)
        large = _records([(0.05 * (i + 1), 1.0 if i >= 10 else 0.0) for i in range(20)])
        assert fit_calibration_artifact(large) == fit_calibration_artifact(large)

    def test_identity_artifact_roundtrips(self) -> None:
        records = _records([(0.6, 1.0), (0.4, 0.0), (0.7, 1.0)])
        artifact = fit_calibration_artifact(records)
        restored = CalibrationArtifact.from_dict(artifact.to_dict())
        assert restored.recalibrator.apply(0.42) == pytest.approx(0.42)
        assert restored.alpha == pytest.approx(artifact.alpha)
