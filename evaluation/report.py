"""Evaluation report assembly (C6.8 support).

Ties the proper scores, per-domain breakdown, mandatory baseline deltas,
reliability diagram, and leakage audit into one rendered report — the shape
CLAUDE.md §2.3/§10 requires (never a bare score). The CLI renders this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    # How the scored probabilities were calibrated (method, n, alpha, floor,
    # fallback flag, artifact hash) — a fallback run must be visibly labeled.
    calibration_provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EvalContext:
    """Everything the ``delphi eval`` command needs for one suite."""

    inputs: EvalInputs
    harness: EvalHarness
    judge: LeakageJudge | None = None


def render_report(
    inputs: EvalInputs,
    *,
    harness: EvalHarness,
    seed: int = 0,
    judge: LeakageJudge | None = None,
) -> str:
    """Score the suite through the guarded harness and render the full report.

    Leakage-first (§2.6): every rendered report carries a leakage-audit section
    — the real audit when a ``judge`` and traces are available, an explicit
    NOT-RUN warning otherwise. A retrospective score with no leakage section is
    exactly the silent failure mode the directive forbids.
    """
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

    if inputs.calibration_provenance is not None:
        prov = inputs.calibration_provenance
        lines.append("\n## Calibration provenance")
        keys = (
            "recalibrator",
            "n",
            "alpha",
            "floor",
            "artifact_hash",
            "source",
            "excluded_fit_overlap",
        )
        for key in keys:
            if key in prov:
                lines.append(f"- {key}: {prov[key]}")
        if prov.get("fallback"):
            lines.append(
                "- WARNING: identity FALLBACK — the calibration fit set was too "
                "small to trust; scored probabilities are the raw pass-through."
            )

    lines.append("\n## Leakage audit (§2.6)")
    if judge is not None and inputs.traces:
        lines.append(render_leakage_audit(judge, inputs.traces))
    else:
        reason = "no leakage judge configured" if judge is None else "no traces collected"
        lines.append(
            f"- WARNING: NOT RUN ({reason}) — the scores above are suspect "
            "until leakage-audited; a great score on a leaky benchmark is noise."
        )

    snapshot = harness.ledger_snapshot()
    lines.append("\n## Trials ledger (§2.4)")
    lines.append(
        f"- debited: {snapshot.debited} / cap {snapshot.cap} "
        f"(outstanding reserved: {snapshot.outstanding_reserved})"
    )
    if not harness.ledger_durable:
        lines.append(
            "- WARNING: EPHEMERAL ledger — this run's draws do not persist, so "
            "the global append-only trials count is NOT being enforced across runs."
        )
    return "\n".join(lines)


def render_leakage_audit(judge: LeakageJudge, traces: Sequence[Trace]) -> str:
    """Render a leakage audit for a suite's traces (C6.8.b)."""
    if not traces:
        return "leakage audit: no traces to audit."
    return audit_suite(judge, traces).render()
