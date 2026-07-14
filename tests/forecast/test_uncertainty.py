"""Unit tests for forecast uncertainty decomposition (UQ1-UQ6 + §8)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.forecast.calibration import calibrate_ensemble
from core.forecast.ensemble import EnsembleForecast, build_ensemble
from core.forecast.llm import ForecastDraw
from core.forecast.uncertainty import (
    DEFAULT_HAIRCUT_SENSITIVITY,
    UncertaintyConfig,
    apply_stability_haircut,
    combine_uncertainty,
    event_uncertainty,
    quantify_from_calibrated,
    quantify_from_ensemble,
    quantify_uncertainty,
    stability_haircut,
)


def _draw(probability: float, run_index: int = 0) -> ForecastDraw:
    return ForecastDraw(
        probability=probability,
        run_index=run_index,
        model_version="m1",
        prompt_version="p1",
        provenance={"run_index": run_index},
    )


def _ensemble(probabilities: list[float]) -> EnsembleForecast:
    draws = [_draw(p, i) for i, p in enumerate(probabilities)]
    return build_ensemble(
        draws,
        aggregator="median",
        knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
    )


class TestDecomposition:
    """UQ1: event vs LLM-output uncertainty are not conflated."""

    def test_confident_event_high_spread(self) -> None:
        u = quantify_uncertainty(0.999, 0.25)
        assert u.event_uncertainty < 0.05
        assert u.llm_output_uncertainty == pytest.approx(0.25)
        assert u.llm_output_uncertainty > u.event_uncertainty

    def test_uncertain_event_low_spread(self) -> None:
        u = quantify_uncertainty(0.5, 0.02)
        assert u.event_uncertainty == pytest.approx(0.5)
        assert u.llm_output_uncertainty == pytest.approx(0.02)
        assert u.event_uncertainty > u.llm_output_uncertainty


class TestLlmOutputFromSpread:
    """UQ2: high ensemble spread → high LLM-output uncertainty."""

    def test_high_spread(self) -> None:
        u_high = quantify_uncertainty(0.6, 0.20)
        u_low = quantify_uncertainty(0.6, 0.01)
        assert u_high.llm_output_uncertainty > u_low.llm_output_uncertainty

    def test_zero_spread(self) -> None:
        u = quantify_uncertainty(0.6, 0.0)
        assert u.llm_output_uncertainty == 0.0
        assert u.stability_haircut == pytest.approx(1.0)


class TestSizingHaircut:
    """UQ3: LLM-output instability reduces sizing input, independent of event-uncertainty."""

    def test_high_instability_reduces_exposure(self) -> None:
        stable = quantify_uncertainty(0.6, 0.01)
        unstable = quantify_uncertainty(0.6, 0.25)
        base_exposure = 1.0
        assert apply_stability_haircut(base_exposure, unstable) < apply_stability_haircut(
            base_exposure, stable
        )

    def test_event_uncertainty_does_not_change_haircut(self) -> None:
        low_p = quantify_uncertainty(0.99, 0.15)
        high_p = quantify_uncertainty(0.51, 0.15)
        assert low_p.stability_haircut == pytest.approx(high_p.stability_haircut)
        assert low_p.event_uncertainty != pytest.approx(high_p.event_uncertainty)

    def test_haircut_formula(self) -> None:
        llm = 0.15
        expected = 1.0 / (1.0 + DEFAULT_HAIRCUT_SENSITIVITY * llm)
        assert stability_haircut(llm) == pytest.approx(expected)


class TestCombination:
    """UQ4: combined uncertainty is monotone in each source."""

    def test_monotone_in_event(self) -> None:
        llm = 0.1
        low = quantify_uncertainty(0.2, llm)
        high = quantify_uncertainty(0.5, llm)
        assert high.combined > low.combined
        assert high.llm_output_uncertainty == pytest.approx(low.llm_output_uncertainty)

    def test_monotone_in_llm(self) -> None:
        p = 0.6
        low = quantify_uncertainty(p, 0.02)
        high = quantify_uncertainty(p, 0.20)
        assert high.combined > low.combined
        assert high.event_uncertainty == pytest.approx(low.event_uncertainty)

    def test_variance_addition(self) -> None:
        event = event_uncertainty(0.7)
        llm = 0.12
        expected = combine_uncertainty(event, llm)
        u = quantify_uncertainty(0.7, llm)
        assert u.combined == pytest.approx(expected)


class TestDeterminism:
    """UQ6: same inputs → same outputs."""

    def test_quantify_is_deterministic(self) -> None:
        a = quantify_uncertainty(0.65, 0.08)
        b = quantify_uncertainty(0.65, 0.08)
        assert a == b

    def test_from_ensemble_is_deterministic(self) -> None:
        ens = _ensemble([0.6, 0.62, 0.58, 0.61, 0.59])
        assert quantify_from_ensemble(ens) == quantify_from_ensemble(ens)


class TestSectionEight:
    """§8: happy path, boundaries, failure modes."""

    def test_happy_path_provenance(self) -> None:
        u = quantify_uncertainty(0.7, 0.1)
        assert u.provenance["event_metric"] == "bernoulli_std"
        assert u.provenance["combination"] == "variance_addition"

    def test_boundary_p_at_zero_and_one(self) -> None:
        assert event_uncertainty(0.0) == pytest.approx(0.0)
        assert event_uncertainty(1.0) == pytest.approx(0.0)

    def test_from_calibrated_adapter(self) -> None:
        ens = _ensemble([0.55, 0.57, 0.53])
        calibrated = calibrate_ensemble(ens)
        u = quantify_from_calibrated(calibrated)
        assert u.llm_output_uncertainty == pytest.approx(calibrated.ensemble_uncertainty)
        assert u.event_uncertainty == pytest.approx(
            event_uncertainty(calibrated.calibrated_probability)
        )

    def test_invalid_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            quantify_uncertainty(1.5, 0.1)
        with pytest.raises(ValueError, match="probability"):
            event_uncertainty(-0.1)

    def test_negative_spread_raises(self) -> None:
        with pytest.raises(ValueError, match="ensemble_spread"):
            quantify_uncertainty(0.5, -0.01)

    def test_invalid_exposure_raises(self) -> None:
        u = quantify_uncertainty(0.5, 0.1)
        with pytest.raises(ValueError, match="exposure"):
            apply_stability_haircut(float("nan"), u)

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError, match="haircut_sensitivity"):
            UncertaintyConfig(haircut_sensitivity=0.0)

    def test_custom_config_changes_haircut(self) -> None:
        cfg_tight = UncertaintyConfig(haircut_sensitivity=10.0)
        cfg_loose = UncertaintyConfig(haircut_sensitivity=2.0)
        u_tight = quantify_uncertainty(0.6, 0.15, config=cfg_tight)
        u_loose = quantify_uncertainty(0.6, 0.15, config=cfg_loose)
        assert u_tight.stability_haircut < u_loose.stability_haircut
