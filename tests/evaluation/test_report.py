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


def test_render_report_calibration_provenance_section() -> None:
    inputs = EvalInputs(
        records=_records(),
        calibration_provenance={
            "recalibrator": "platt",
            "n": 40,
            "alpha": 1.25,
            "floor": 0.01,
            "artifact_hash": "abc123",
            "fallback": False,
        },
    )
    rendered = render_report(inputs, harness=_harness())
    assert "## Calibration provenance" in rendered
    assert "- recalibrator: platt" in rendered
    assert "- artifact_hash: abc123" in rendered
    assert "FALLBACK" not in rendered


def test_render_report_shows_fit_overlap_exclusions() -> None:
    inputs = EvalInputs(
        records=_records(),
        calibration_provenance={"recalibrator": "platt", "excluded_fit_overlap": 7},
    )
    rendered = render_report(inputs, harness=_harness())
    assert "- excluded_fit_overlap: 7" in rendered


def test_render_report_flags_fallback_calibration() -> None:
    inputs = EvalInputs(
        records=_records(),
        calibration_provenance={"recalibrator": "platt", "n": 3, "fallback": True},
    )
    rendered = render_report(inputs, harness=_harness())
    assert "identity FALLBACK" in rendered


def test_render_report_omits_provenance_when_absent() -> None:
    rendered = render_report(EvalInputs(records=_records()), harness=_harness())
    assert "Calibration provenance" not in rendered


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


def test_render_report_includes_leakage_audit_with_judge_and_traces() -> None:
    judge = LeakageJudge(
        FixtureLeakageJudgeLLM(flag_substrings=("LEAK",), reject_future_iso_dates=False)
    )
    inputs = EvalInputs(
        records=_records(),
        traces=(
            Trace(component=TraceComponent.SEARCH, as_of=_AS_OF, text="LEAK"),
            Trace(component=TraceComponent.SEARCH, as_of=_AS_OF, text="clean"),
        ),
    )
    rendered = render_report(inputs, harness=_harness(), judge=judge)
    assert "## Leakage audit (§2.6)" in rendered
    assert "leakage_rate=0.5000" in rendered
    assert "NOT RUN" not in rendered


def test_render_report_warns_loudly_without_judge() -> None:
    rendered = render_report(EvalInputs(records=_records()), harness=_harness())
    assert "## Leakage audit (§2.6)" in rendered
    assert "NOT RUN (no leakage judge configured)" in rendered
    assert "suspect" in rendered


def test_render_report_warns_loudly_without_traces() -> None:
    judge = LeakageJudge(FixtureLeakageJudgeLLM())
    rendered = render_report(EvalInputs(records=_records()), harness=_harness(), judge=judge)
    assert "NOT RUN (no traces collected)" in rendered


def test_render_report_shows_trials_ledger_state() -> None:
    # Self-counting ledger (no injected trials_count): the report's own guarded
    # draw is visible — 3 records debited against cap 100.
    harness = EvalHarness(budget_ledger=InMemoryBudgetLedger(cap=100))
    rendered = render_report(EvalInputs(records=_records()), harness=harness)
    assert "## Trials ledger (§2.4)" in rendered
    assert "- debited: 3 / cap 100 (outstanding reserved: 0)" in rendered


def test_render_report_warns_on_ephemeral_ledger() -> None:
    rendered = render_report(EvalInputs(records=_records()), harness=_harness())
    assert "EPHEMERAL ledger" in rendered


def test_render_report_no_ephemeral_warning_on_durable_ledger() -> None:
    class DurableLedger(InMemoryBudgetLedger):
        @property
        def durable(self) -> bool:
            return True

    harness = EvalHarness(budget_ledger=DurableLedger(cap=100))
    rendered = render_report(EvalInputs(records=_records()), harness=harness)
    assert "## Trials ledger (§2.4)" in rendered
    assert "EPHEMERAL" not in rendered
