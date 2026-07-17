"""Unit tests for forecast ensemble math (§8)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from core.forecast.ensemble import (
    aggregate,
    build_ensemble,
    build_ensemble_config,
    ensemble_std,
)
from core.forecast.llm import ForecastDraw


def _draw(probability: float, run_index: int = 0) -> ForecastDraw:
    return ForecastDraw(
        probability=probability,
        run_index=run_index,
        model_version="m1",
        prompt_version="p1",
        provenance={"run_index": run_index},
    )


class TestAggregate:
    def test_happy_path_median(self) -> None:
        assert aggregate([0.2, 0.5, 0.8], "median") == pytest.approx(0.5)

    def test_happy_path_trimmed_mean(self) -> None:
        result = aggregate([0.1, 0.4, 0.5, 0.6, 0.9], "trimmed_mean", trim_fraction=0.2)
        assert 0.4 <= result <= 0.6

    def test_boundary_trimmed_mean_small_n_falls_back_to_mean(self) -> None:
        assert aggregate([0.2, 0.8], "trimmed_mean") == pytest.approx(0.5)

    def test_failure_empty_probabilities_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            aggregate([], "median")

    def test_failure_invalid_aggregator_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            aggregate([0.5], "mean")  # type: ignore[arg-type]


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class TestLogOddsAggregators:
    """§3.5: combination happens in log-odds space."""

    def test_mean_matches_closed_form(self) -> None:
        probs = [0.2, 0.5, 0.8]
        expected = _sigmoid(sum(_logit(p) for p in probs) / 3)
        assert aggregate(probs, "log_odds_mean") == pytest.approx(expected)

    def test_symmetry_around_half(self) -> None:
        probs = [0.15, 0.4, 0.7, 0.9]
        flipped = [1.0 - p for p in probs]
        assert aggregate(probs, "log_odds_mean") == pytest.approx(
            1.0 - aggregate(flipped, "log_odds_mean")
        )
        assert aggregate(probs, "log_odds_trimmed_mean") == pytest.approx(
            1.0 - aggregate(flipped, "log_odds_trimmed_mean")
        )

    def test_single_draw_is_identity(self) -> None:
        assert aggregate([0.37], "log_odds_mean") == pytest.approx(0.37)
        assert aggregate([0.37], "log_odds_trimmed_mean") == pytest.approx(0.37)

    def test_identical_draws_are_fixed_point(self) -> None:
        assert aggregate([0.6, 0.6, 0.6, 0.6], "log_odds_trimmed_mean") == pytest.approx(0.6)

    def test_zero_and_one_draws_are_clamped_not_dominant(self) -> None:
        # A 0.0 draw maps to logit(1e-3), not -inf: the pool stays finite and
        # the other draws still matter.
        result = aggregate([0.0, 0.5, 0.5], "log_odds_mean")
        assert 0.0 < result < 0.5
        assert result > 0.05
        high = aggregate([1.0, 0.5, 0.5], "log_odds_mean")
        assert high == pytest.approx(1.0 - result)

    def test_pooling_amplifies_agreement_beyond_probability_mean(self) -> None:
        # Agreeing confident draws pool more extreme than the arithmetic mean —
        # the whole point of log-odds pooling.
        probs = [0.9, 0.8, 0.85]
        assert aggregate(probs, "log_odds_mean") > 0.8

    def test_trimmed_variant_trims_in_logit_space(self) -> None:
        # An extreme outlier at 0.999 is dropped by the trim: the trimmed pool
        # must be closer to the crowd than the untrimmed pool.
        probs = [0.4, 0.45, 0.5, 0.55, 0.999]
        trimmed = aggregate(probs, "log_odds_trimmed_mean", trim_fraction=0.2)
        untrimmed = aggregate(probs, "log_odds_mean")
        assert trimmed < untrimmed
        assert trimmed == pytest.approx(_sigmoid(sum(_logit(p) for p in [0.45, 0.5, 0.55]) / 3))

    def test_custom_pool_eps(self) -> None:
        tight = aggregate([0.0, 0.5], "log_odds_mean", pool_eps=1e-6)
        loose = aggregate([0.0, 0.5], "log_odds_mean", pool_eps=0.1)
        assert tight < loose

    @pytest.mark.parametrize("pool_eps", [0.0, -0.1, 0.5, 0.7])
    def test_invalid_pool_eps_raises(self, pool_eps: float) -> None:
        with pytest.raises(ValueError, match="pool_eps"):
            aggregate([0.5], "log_odds_mean", pool_eps=pool_eps)


class TestEnsembleStd:
    def test_happy_path_sample_std(self) -> None:
        assert ensemble_std([0.4, 0.6]) == pytest.approx(np.std([0.4, 0.6], ddof=1))

    def test_boundary_single_value_is_zero(self) -> None:
        assert ensemble_std([0.7]) == 0.0

    def test_boundary_empty_is_zero(self) -> None:
        assert ensemble_std([]) == 0.0


class TestBuildEnsemble:
    def test_happy_path_builds_forecast(self) -> None:
        draws = (_draw(0.4, 0), _draw(0.6, 1))
        kt = datetime(2024, 1, 1, tzinfo=UTC)
        result = build_ensemble(draws, aggregator="median", knowledge_time=kt)
        assert result.probability == pytest.approx(0.5)
        assert result.uncertainty == pytest.approx(ensemble_std([0.4, 0.6]))
        assert result.n == 2
        assert result.provenance["aggregation_method"] == "median"

    def test_failure_empty_draws_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_ensemble([], aggregator="median", knowledge_time=datetime.now(tz=UTC))


class TestBuildEnsembleConfig:
    def test_happy_path_serializes_config(self) -> None:
        cfg = build_ensemble_config(n=10, aggregator="median")
        assert cfg == "n=10|agg=median|trim=0.1|spread=std"

    def test_boundary_custom_trim(self) -> None:
        cfg = build_ensemble_config(n=5, aggregator="trimmed_mean", trim_fraction=0.2)
        assert "trim=0.2" in cfg

    def test_legacy_aggregator_config_is_unchanged(self) -> None:
        # Cache-key regression guard: existing median/trimmed_mean keys must not
        # move when the log-odds aggregators are introduced.
        assert build_ensemble_config(n=4, aggregator="median") == (
            "n=4|agg=median|trim=0.1|spread=std"
        )

    def test_log_odds_config_carries_pool_eps(self) -> None:
        cfg = build_ensemble_config(n=12, aggregator="log_odds_trimmed_mean")
        assert cfg == "n=12|agg=log_odds_trimmed_mean|trim=0.1|spread=std|pool_eps=0.001"


class TestBuildEnsembleLogOdds:
    def test_provenance_records_pool_eps(self) -> None:
        draws = (_draw(0.4, 0), _draw(0.6, 1))
        kt = datetime(2024, 1, 1, tzinfo=UTC)
        result = build_ensemble(draws, aggregator="log_odds_mean", knowledge_time=kt)
        assert result.probability == pytest.approx(0.5)
        assert result.provenance["pool_eps"] == 0.001
        assert result.provenance["aggregation_method"] == "log_odds_mean"

    def test_legacy_provenance_has_no_pool_eps(self) -> None:
        draws = (_draw(0.4, 0), _draw(0.6, 1))
        kt = datetime(2024, 1, 1, tzinfo=UTC)
        result = build_ensemble(draws, aggregator="median", knowledge_time=kt)
        assert "pool_eps" not in result.provenance
