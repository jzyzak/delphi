"""Tests for the evaluation report assembly (C6.8 support)."""

from __future__ import annotations

from datetime import UTC, datetime

from core.forecast.leakage_judge import (
    FixtureLeakageJudgeLLM,
    LeakageJudge,
    Trace,
    TraceComponent,
)
from core.orchestration.budget import InMemoryBudgetLedger
from evaluation.baselines import Baseline
from evaluation.harness import EvalHarness
from evaluation.report import EvalInputs, render_leakage_audit, render_report
from evaluation.scoring import ScoredRecord

_AS_OF = datetime(2026, 1, 1, tzinfo=UTC)


def _records() -> tuple[ScoredRecord, ...]:
    return (
        ScoredRecord(question_id="q1", domain="econ", probability=0.8, outcome=1.0),
        ScoredRecord(question_id="q2", domain="econ", probability=0.2, outcome=0.0),
        ScoredRecord(question_id="q3", domain="geo", probability=0.6, outcome=1.0),
    )


def _harness() -> EvalHarness:
    return EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=100, trials_count=lambda: 0))


def test_render_report_sections() -> None:
    inputs = EvalInputs(
        records=_records(),
        baselines=(Baseline(name="market", predictions={"q1": 0.5, "q2": 0.5, "q3": 0.5}),),
    )
    rendered = render_report(inputs, harness=_harness())
    assert "Proper scores" in rendered
    assert "brier" in rendered
    assert "Per-domain" in rendered
    assert "Baseline deltas" in rendered
    assert "vs market" in rendered
    assert "Reliability" in rendered
    assert "ECE=" in rendered


def test_render_report_without_baselines() -> None:
    rendered = render_report(EvalInputs(records=_records()), harness=_harness())
    assert "Baseline deltas" not in rendered


def test_render_leakage_audit() -> None:
    judge = LeakageJudge(
        FixtureLeakageJudgeLLM(flag_substrings=("LEAK",), reject_future_iso_dates=False)
    )
    traces = (Trace(component=TraceComponent.SEARCH, as_of=_AS_OF, text="LEAK"),)
    assert "leakage_rate" in render_leakage_audit(judge, traces)


def test_render_leakage_audit_no_traces() -> None:
    judge = LeakageJudge(FixtureLeakageJudgeLLM())
    assert "no traces" in render_leakage_audit(judge, ())
