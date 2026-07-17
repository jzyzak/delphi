"""Unit tests for the leakage-judge gate (C4.6)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from core.forecast.ensemble import EnsembleForecast, build_ensemble
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge, TraceComponent
from core.forecast.llm import ForecastDraw
from core.forecast.search import Evidence
from core.forecast.supervisor import Confidence, DisagreementKind, ReconciledForecast
from forecaster.stages.leakage_gate import run_leakage_gate

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _evidence(snippet: str) -> Evidence:
    return Evidence(
        snippet=snippet,
        source="tavily",
        source_id="http://item",
        knowledge_time=datetime(2024, 5, 1, tzinfo=UTC),
        score=0.9,
    )


def _ensemble(probs: Sequence[float]) -> EnsembleForecast:
    draws = tuple(
        ForecastDraw(probability=p, run_index=i, model_version="m", prompt_version="pv")
        for i, p in enumerate(probs)
    )
    return build_ensemble(draws, aggregator="median", knowledge_time=AS_OF)


def _reconciled(prob: float) -> ReconciledForecast:
    return ReconciledForecast(
        probability=prob,
        uncertainty=0.05,
        aggregate_probability=prob,
        confidence=Confidence.LOW,
        applied=False,
        knowledge_time=AS_OF,
        disagreement=DisagreementKind.NONE,
    )


def test_clean_trace_not_flagged() -> None:
    judge = LeakageJudge(FixtureLeakageJudgeLLM())
    result = run_leakage_gate(_ensemble([0.6, 0.6]), _reconciled(0.6), judge=judge)
    assert result.flagged is False
    assert result.quarantine == ()
    assert len(result.verdicts) == 2


def test_planted_leak_is_flagged_and_quarantined() -> None:
    # The ensemble trace always contains "aggregation_method" in its provenance.
    judge = LeakageJudge(FixtureLeakageJudgeLLM(flag_substrings=("aggregation_method",)))
    result = run_leakage_gate(
        _ensemble([0.6, 0.6]), _reconciled(0.6), judge=judge, forecast_id="q-1"
    )
    assert result.flagged is True
    assert len(result.quarantine) >= 1
    assert result.quarantine[0].forecast_id == "q-1"


def test_evidence_adds_search_trace_audited_first() -> None:
    judge = LeakageJudge(FixtureLeakageJudgeLLM())
    result = run_leakage_gate(
        _ensemble([0.6, 0.6]),
        _reconciled(0.6),
        judge=judge,
        evidence=(_evidence("clean pre-as-of snippet"),),
    )
    assert result.flagged is False
    assert len(result.verdicts) == 3
    assert result.verdicts[0].component is TraceComponent.SEARCH


def test_leak_in_evidence_snippet_is_flagged_and_quarantined() -> None:
    # A retrieval-side leak lives in the raw snippet text — the numeric
    # ensemble/supervisor traces never see it (§2.1 defense-in-depth).
    judge = LeakageJudge(FixtureLeakageJudgeLLM(flag_substrings=("the election was won by",)))
    result = run_leakage_gate(
        _ensemble([0.6, 0.6]),
        _reconciled(0.6),
        judge=judge,
        forecast_id="q-2",
        evidence=(_evidence("... the election was won by X ..."),),
    )
    assert result.flagged is True
    assert result.quarantine[0].component is TraceComponent.SEARCH
    assert result.quarantine[0].forecast_id == "q-2"


def test_no_evidence_means_no_search_trace() -> None:
    judge = LeakageJudge(FixtureLeakageJudgeLLM())
    result = run_leakage_gate(_ensemble([0.6, 0.6]), _reconciled(0.6), judge=judge, evidence=())
    assert len(result.verdicts) == 2
    assert all(v.component is not TraceComponent.SEARCH for v in result.verdicts)
