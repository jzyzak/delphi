"""Unit tests for the aggregate + supervisor reconcile stage (C4.4)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from core.forecast.ensemble import EnsembleForecast, build_ensemble
from core.forecast.llm import ForecastDraw
from core.forecast.search import FixtureAsOfSearch
from core.forecast.supervisor import (
    Confidence,
    FixtureSupervisorLLM,
    FixtureSupervisorResponse,
)
from forecaster.stages.aggregate import reconcile

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _ensemble(probs: Sequence[float], aggregator: str = "median") -> EnsembleForecast:
    draws = tuple(
        ForecastDraw(probability=p, run_index=i, model_version="m", prompt_version="pv")
        for i, p in enumerate(probs)
    )
    return build_ensemble(draws, aggregator=aggregator, knowledge_time=AS_OF)  # type: ignore[arg-type]


def test_no_disagreement_falls_back_to_aggregate() -> None:
    ensemble = _ensemble([0.7, 0.7, 0.7, 0.7])
    result = reconcile(
        ensemble, searcher=FixtureAsOfSearch(), supervisor_llm=FixtureSupervisorLLM()
    )
    assert result.applied is False
    assert result.probability == pytest.approx(ensemble.probability)


def test_high_confidence_reconciliation_is_applied() -> None:
    ensemble = _ensemble([0.1, 0.1, 0.9, 0.9])  # multimodal disagreement
    supervisor = FixtureSupervisorLLM(
        {"multimodal": FixtureSupervisorResponse(probability=0.8, confidence=Confidence.HIGH)}
    )
    result = reconcile(ensemble, searcher=FixtureAsOfSearch(), supervisor_llm=supervisor)
    assert result.applied is True
    assert result.probability == pytest.approx(0.8)


def test_low_confidence_falls_back() -> None:
    ensemble = _ensemble([0.1, 0.1, 0.9, 0.9])
    supervisor = FixtureSupervisorLLM(
        {"multimodal": FixtureSupervisorResponse(probability=0.8, confidence=Confidence.LOW)}
    )
    result = reconcile(ensemble, searcher=FixtureAsOfSearch(), supervisor_llm=supervisor)
    assert result.applied is False
    assert result.probability == pytest.approx(ensemble.probability)


class TestSupervisorTuning:
    def test_medium_confidence_applied_when_gate_relaxed(self) -> None:
        from forecaster.stages.aggregate import SupervisorTuning, reconcile

        ensemble = _ensemble([0.1, 0.1, 0.9, 0.9])  # multimodal disagreement
        supervisor = FixtureSupervisorLLM(
            {"multimodal": FixtureSupervisorResponse(probability=0.8, confidence=Confidence.MEDIUM)}
        )
        tuned = reconcile(
            ensemble,
            searcher=FixtureAsOfSearch(),
            supervisor_llm=supervisor,
            tuning=SupervisorTuning(min_apply_confidence=Confidence.MEDIUM),
        )
        assert tuned.applied is True
        assert tuned.probability == 0.8

    def test_medium_confidence_falls_back_at_default_gate(self) -> None:
        from forecaster.stages.aggregate import reconcile

        ensemble = _ensemble([0.1, 0.1, 0.9, 0.9])
        supervisor = FixtureSupervisorLLM(
            {"multimodal": FixtureSupervisorResponse(probability=0.8, confidence=Confidence.MEDIUM)}
        )
        result = reconcile(ensemble, searcher=FixtureAsOfSearch(), supervisor_llm=supervisor)
        assert result.applied is False
        assert result.probability == ensemble.probability

    def test_spread_threshold_zero_triggers_on_small_spread(self) -> None:
        from forecaster.stages.aggregate import SupervisorTuning, reconcile

        ensemble = _ensemble([0.48, 0.5, 0.52])  # tiny spread; default never triggers
        supervisor = FixtureSupervisorLLM(
            default=FixtureSupervisorResponse(probability=0.7, confidence=Confidence.HIGH)
        )
        result = reconcile(
            ensemble,
            searcher=FixtureAsOfSearch(),
            supervisor_llm=supervisor,
            tuning=SupervisorTuning(spread_threshold=0.0),
        )
        assert supervisor.call_count == 1  # supervisor consulted despite tiny spread
        assert result.applied is True
