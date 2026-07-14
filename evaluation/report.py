"""Evaluation report assembly (C6.8 support).

Ties the proper scores, per-domain breakdown, mandatory baseline deltas,
reliability diagram, and leakage audit into one rendered report — the shape
CLAUDE.md §2.3/§10 requires (never a bare score). The CLI renders this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from core.forecast.leakage_judge import LeakageJudge, Trace
from evaluation.aggregate import baseline_delta, per_domain_summary
from evaluation.baselines import Baseline
from evaluation.harness import EvalHarness
from evaluation.leakage_audit import audit_suite
from evaluation.reliability import reliability
from evaluation.scoring import BrierScorer, LogScorer, ScoredRecord, Scorer

__all__ = ["EvalContext", "EvalInputs", "render_leakage_audit", "render_report"]


@dataclass(frozen=True)
class EvalInputs:
    """The scored records, baselines, and traces for one evaluation suite."""

    records: tuple[ScoredRecord, ...]
    baselines: tuple[Baseline, ...] = ()
    traces: tuple[Trace, ...] = ()
    scorers: tuple[Scorer, ...] = field(default_factory=lambda: (BrierScorer(), LogScorer()))


@dataclass(frozen=True)
class EvalContext:
    """Everything the ``delphi eval`` command needs for one suite."""

    inputs: EvalInputs
    harness: EvalHarness
    judge: LeakageJudge | None = None


def render_report(inputs: EvalInputs, *, harness: EvalHarness, seed: int = 0) -> str:
    """Score the suite through the guarded harness and render the full report."""
    scorers = inputs.scorers
    summaries = harness.evaluate_guarded_scorers(scorers, inputs.records, seed=seed)

    lines = [f"# Evaluation ({len(inputs.records)} questions)", "", "## Proper scores"]
    for name, summary in summaries.items():
        lines.append(
            f"- {name}: {summary.mean:.4f} (95% CI [{summary.ci_low:.4f}, {summary.ci_high:.4f}])"
        )

    brier = BrierScorer()
    lines.append("\n## Per-domain (brier)")
    for domain, summary in per_domain_summary(brier, inputs.records, seed=seed).items():
        lines.append(f"- {domain}: {summary.mean:.4f} (n={summary.n})")

    if inputs.baselines:
        lines.append("\n## Baseline deltas (brier; negative = model beats baseline)")
        for baseline in inputs.baselines:
            delta = baseline_delta(brier, inputs.records, baseline)
            rendered = "n/a" if delta is None else f"{delta:+.4f}"
            lines.append(f"- vs {baseline.name}: {rendered}")

    lines.append("\n## Reliability")
    diagram = reliability(
        [r.probability for r in inputs.records],
        [r.outcome for r in inputs.records],
    )
    lines.append(diagram.render())
    return "\n".join(lines)


def render_leakage_audit(judge: LeakageJudge, traces: Sequence[Trace]) -> str:
    """Render a leakage audit for a suite's traces (C6.8.b)."""
    if not traces:
        return "leakage audit: no traces to audit."
    return audit_suite(judge, traces).render()
