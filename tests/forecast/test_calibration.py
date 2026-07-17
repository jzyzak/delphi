"""Unit tests for forecast calibration (CA1-CA7 + §8)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from core.forecast.calibration import (
    DEFAULT_ALPHA,
    DEFAULT_BOUNDARY_MARGIN,
    DEFAULT_SPREAD_THRESHOLD,
    FrozenCalibration,
    apply_floor,
    calibrate,
    calibrate_ensemble,
    near_decision_boundary,
)
from core.forecast.ensemble import build_ensemble
from core.forecast.llm import ForecastDraw


def _draw(probability: float, run_index: int = 0) -> ForecastDraw:
    return ForecastDraw(
        probability=probability,
        run_index=run_index,
        model_version="m1",
        prompt_version="p1",
        provenance={"run_index": run_index},
    )


def _geometric_mean_platt(p: float, alpha: float) -> float:
    """Closed-form geometric-mean Platt equivalent: p^a / (p^a + (1-p)^a)."""
    p_alpha = p**alpha
    q_alpha = (1.0 - p) ** alpha
    return p_alpha / (p_alpha + q_alpha)


class TestCalibrateDirection:
    """CA1: calibration pushes p away from 0.5 toward 0/1; monotonic."""

    def test_happy_path_pushes_above_half_up(self) -> None:
        assert calibrate(0.6) > 0.6

    def test_happy_path_pushes_below_half_down(self) -> None:
        assert calibrate(0.4) < 0.4

    def test_boundary_half_is_identity(self) -> None:
        assert calibrate(0.5) == pytest.approx(0.5)

    def test_monotonic_increasing(self) -> None:
        probs = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
        calibrated = [calibrate(p) for p in probs]
        assert calibrated == sorted(calibrated)


class TestFixedCoefficient:
    """CA2: default is sqrt(3); no code path fits alpha on outcome data."""

    def test_happy_path_default_alpha_is_sqrt_three(self) -> None:
        assert pytest.approx(math.sqrt(3)) == DEFAULT_ALPHA

    def test_default_matches_explicit_sqrt_three(self) -> None:
        assert calibrate(0.6) == pytest.approx(calibrate(0.6, alpha=math.sqrt(3)))

    def test_no_fit_or_train_api_surface(self) -> None:
        import core.forecast.calibration as calibration_module

        module_attrs = set(dir(calibration_module))
        forbidden = {
            name
            for name in module_attrs
            if any(k in name.lower() for k in ("fit", "train", "learn"))
        }
        assert forbidden == set()
        assert not hasattr(calibration_module, "fit_calibration")
        assert not hasattr(calibration_module, "train_calibration")


class TestWrongSideCaveat:
    """CA3: wrong-side forecasts amplified wrong; diagnostic flags risky cases."""

    def test_wrong_side_pushed_further_wrong(self) -> None:
        # Event succeeds but forecast is below 0.5 — extremization makes it worse.
        assert calibrate(0.4) < 0.4

    def test_diagnostic_flags_near_half(self) -> None:
        assert near_decision_boundary(0.52, 0.01) is True

    def test_diagnostic_flags_high_spread(self) -> None:
        assert near_decision_boundary(0.7, 0.2) is True

    def test_diagnostic_clear_when_confident_and_decisive(self) -> None:
        assert near_decision_boundary(0.75, 0.05) is False


class TestCalibrateMath:
    """CA4: sigmoid(alpha*logit(p)) matches geometric-mean Platt form."""

    def test_happy_path_matches_geometric_mean_platt(self) -> None:
        p = 0.6
        alpha = math.sqrt(3)
        assert calibrate(p, alpha=alpha) == pytest.approx(_geometric_mean_platt(p, alpha))

    def test_boundary_extremes_match_closed_form(self) -> None:
        alpha = DEFAULT_ALPHA
        for p in (0.01, 0.25, 0.75, 0.99):
            assert calibrate(p, alpha=alpha) == pytest.approx(_geometric_mean_platt(p, alpha))


class TestStabilityDeterminism:
    """CA5: same input -> same output; stable at p near 0 and 1."""

    def test_happy_path_deterministic(self) -> None:
        assert calibrate(0.55) == calibrate(0.55)

    @pytest.mark.parametrize("p", [0.0, 1.0, 1e-15, 1.0 - 1e-15, 1e-6, 1.0 - 1e-6])
    def test_boundary_extremes_finite_and_in_unit_interval(self, p: float) -> None:
        result = calibrate(p)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_boundary_exact_zero_and_one_not_nan(self) -> None:
        low = calibrate(0.0)
        high = calibrate(1.0)
        assert low < 0.01
        assert high > 0.99


class TestNoDoubleExtremization:
    """CA6: uncertainty passed through; only probability calibrated once."""

    def test_happy_path_uncertainty_unchanged(self) -> None:
        draws = (_draw(0.55, 0), _draw(0.65, 1))
        kt = datetime(2024, 1, 1, tzinfo=UTC)
        ensemble = build_ensemble(draws, aggregator="median", knowledge_time=kt)
        result = calibrate_ensemble(ensemble)
        assert result.ensemble_uncertainty == pytest.approx(ensemble.uncertainty)
        assert result.calibrated_probability > ensemble.probability
        assert result.raw_probability == pytest.approx(ensemble.probability)


class TestProvenance:
    """CA7: method + coefficient recorded."""

    def test_happy_path_provenance_records_method_and_alpha(self) -> None:
        draws = (_draw(0.55, 0), _draw(0.65, 1))
        kt = datetime(2024, 1, 1, tzinfo=UTC)
        ensemble = build_ensemble(draws, aggregator="median", knowledge_time=kt)
        result = calibrate_ensemble(ensemble)
        assert result.provenance["calibration_method"] == "platt_logodds_extremization"
        assert result.provenance["alpha"] == pytest.approx(DEFAULT_ALPHA)
        assert result.provenance["boundary_margin"] == pytest.approx(DEFAULT_BOUNDARY_MARGIN)
        assert result.provenance["spread_threshold"] == pytest.approx(DEFAULT_SPREAD_THRESHOLD)


class TestCalibrateFailureModes:
    """§8 failure modes: invalid input must raise."""

    @pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
    def test_failure_invalid_probability_raises(self, p: float) -> None:
        with pytest.raises(ValueError, match="probability"):
            calibrate(p)

    @pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
    def test_failure_invalid_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            calibrate(0.5, alpha=alpha)

    def test_failure_invalid_uncertainty_raises(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            near_decision_boundary(0.5, -0.1)

    def test_failure_invalid_boundary_margin_raises(self) -> None:
        with pytest.raises(ValueError, match="boundary_margin"):
            near_decision_boundary(0.5, 0.1, boundary_margin=-0.01)

    def test_failure_invalid_spread_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="spread_threshold"):
            near_decision_boundary(0.5, 0.1, spread_threshold=-0.01)


class TestApplyFloor:
    """Floor clamp: final probabilities live in [floor, 1 - floor]."""

    def test_none_is_identity(self) -> None:
        assert apply_floor(0.001, None) == 0.001

    def test_clamps_low_tail(self) -> None:
        assert apply_floor(0.001, 0.01) == pytest.approx(0.01)

    def test_clamps_high_tail(self) -> None:
        assert apply_floor(0.999, 0.01) == pytest.approx(0.99)

    def test_interior_untouched(self) -> None:
        assert apply_floor(0.4, 0.01) == 0.4

    @pytest.mark.parametrize("floor", [-0.01, 0.5, 0.7, float("nan")])
    def test_invalid_floor_raises(self, floor: float) -> None:
        with pytest.raises(ValueError, match="floor"):
            apply_floor(0.4, floor)

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            apply_floor(1.5, 0.01)


_ISOTONIC_DICT = {
    "schema_version": 1,
    "method": "isotonic",
    "recalibrator": {"x": [0.1, 0.4, 0.6, 0.9], "y": [0.05, 0.2, 0.7, 0.95], "n": 4},
    "alpha": 1.25,
    "floor": 0.01,
    "n": 4,
    "fitted": {"seed": 0},
}

_PLATT_DICT = {
    "schema_version": 1,
    "method": "platt",
    "recalibrator": {"a": 2.0, "b": 0.5, "n": 12},
    "alpha": 1.0,
    "floor": None,
    "n": 12,
}


class TestFrozenCalibration:
    """The apply-only artifact reader the live chain consumes."""

    def test_isotonic_interpolates_between_knots(self) -> None:
        learned = FrozenCalibration.from_dict(_ISOTONIC_DICT)
        # Midpoint of the (0.4, 0.2) -> (0.6, 0.7) segment.
        assert learned.apply(0.5) == pytest.approx(0.45)

    def test_isotonic_flat_extrapolation_and_clamp(self) -> None:
        learned = FrozenCalibration.from_dict(_ISOTONIC_DICT)
        assert learned.apply(0.0) == pytest.approx(0.05)
        assert learned.apply(1.0) == pytest.approx(0.95)
        clamped = FrozenCalibration.from_dict(
            {
                **_ISOTONIC_DICT,
                "recalibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0], "n": 2},
            }
        )
        assert 0.0 < clamped.apply(0.0) < clamped.apply(1.0) < 1.0

    def test_platt_applies_logistic_map(self) -> None:
        learned = FrozenCalibration.from_dict(_PLATT_DICT)
        logit = math.log(0.7 / 0.3)
        expected = 1.0 / (1.0 + math.exp(-(2.0 * logit + 0.5)))
        assert learned.apply(0.7) == pytest.approx(expected)

    def test_platt_extreme_input_is_clamped(self) -> None:
        learned = FrozenCalibration.from_dict(_PLATT_DICT)
        assert 0.0 < learned.apply(0.0) < learned.apply(1.0) < 1.0

    def test_carries_alpha_floor_and_provenance(self) -> None:
        learned = FrozenCalibration.from_dict(_ISOTONIC_DICT, artifact_hash="abc123")
        assert learned.alpha == 1.25
        assert learned.floor == 0.01
        prov = learned.provenance
        assert prov["recalibrator"] == "isotonic"
        assert prov["fitted"] is True
        assert prov["n"] == 4
        assert prov["artifact_hash"] == "abc123"
        assert prov["seed"] == 0

    def test_fallback_flag_lands_in_provenance(self) -> None:
        learned = FrozenCalibration.from_dict({**_PLATT_DICT, "fallback": True})
        assert learned.provenance["fallback"] is True

    def test_fallback_defaults_false_in_provenance(self) -> None:
        learned = FrozenCalibration.from_dict(_PLATT_DICT)
        assert learned.provenance["fallback"] is False

    def test_pre_schema_version_dict_defaults_to_isotonic(self) -> None:
        legacy = {
            "recalibrator": {"x": [0.1, 0.9], "y": [0.0, 1.0], "n": 2},
            "alpha": 1.5,
        }
        learned = FrozenCalibration.from_dict(legacy)
        assert learned.method == "isotonic"
        assert learned.floor is None

    def test_unsupported_schema_version_raises(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            FrozenCalibration.from_dict({**_ISOTONIC_DICT, "schema_version": 99})

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            FrozenCalibration.from_dict({**_ISOTONIC_DICT, "method": "spline"})

    def test_unknown_method_constructor_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            FrozenCalibration(method="spline", alpha=1.0)

    def test_isotonic_requires_knots(self) -> None:
        with pytest.raises(ValueError, match="knot"):
            FrozenCalibration(method="isotonic", alpha=1.0)

    def test_isotonic_requires_equal_length_knots(self) -> None:
        with pytest.raises(ValueError, match="knot"):
            FrozenCalibration(method="isotonic", alpha=1.0, x_knots=(0.1, 0.9), y_knots=(0.5,))

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            FrozenCalibration.from_dict({**_PLATT_DICT, "alpha": -1.0})

    def test_invalid_floor_raises(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            FrozenCalibration.from_dict({**_PLATT_DICT, "floor": 0.6})

    def test_invalid_probability_raises(self) -> None:
        learned = FrozenCalibration.from_dict(_PLATT_DICT)
        with pytest.raises(ValueError, match="probability"):
            learned.apply(-0.1)

    def test_duplicate_knot_x_uses_first_matching_segment(self) -> None:
        # PAV-fit knots can carry tied x values; an exact-tie query resolves via
        # the first segment whose upper knot bounds it (np.interp agreement is
        # checked by the evaluation-side parity test).
        learned = FrozenCalibration(
            method="isotonic",
            alpha=1.0,
            x_knots=(0.2, 0.5, 0.5, 0.8),
            y_knots=(0.1, 0.3, 0.6, 0.9),
        )
        assert learned.apply(0.5) == pytest.approx(0.3)

    def test_satisfies_recalibrator_protocol(self) -> None:
        from forecaster.stages.calibrate import Recalibrator

        assert isinstance(FrozenCalibration.from_dict(_PLATT_DICT), Recalibrator)
