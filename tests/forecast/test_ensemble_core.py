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
