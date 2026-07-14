"""Holdout-gated rollout (A.4).

The learned conductor replaces the heuristic in production **only** if it beats
it on the guarded set *without* regressing calibration or leakage (CLAUDE.md §4).
Lower proper score is better. If it never beats the heuristic, that is a **fine
outcome** — the heuristic conductor is already the product, so we keep it. This
function is a pure decision over already-computed guarded-set summaries; it never
touches the holdout itself (that is the harness's job, §2.2/§2.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.aggregate import ScoreSummary

__all__ = ["RolloutDecision", "holdout_gated_rollout"]


@dataclass(frozen=True)
class RolloutDecision:
    """Whether to promote the learned conductor, with an auditable reason."""

    promote: bool
    reason: str


def holdout_gated_rollout(
    *,
    learned: ScoreSummary,
    heuristic: ScoreSummary,
    calibration_regressed: bool,
    leakage_regressed: bool,
) -> RolloutDecision:
    """Promote the learned conductor only if it strictly beats the heuristic.

    Calibration or leakage regressions veto promotion regardless of score.
    """
    if calibration_regressed:
        return RolloutDecision(False, "keep heuristic: calibration regressed on the guarded set.")
    if leakage_regressed:
        return RolloutDecision(False, "keep heuristic: leakage regressed on the guarded set.")
    if learned.mean < heuristic.mean:
        return RolloutDecision(
            True,
            f"promote learned: proper score {learned.mean:.4f} beats "
            f"heuristic {heuristic.mean:.4f}.",
        )
    return RolloutDecision(
        False,
        f"keep heuristic: learned {learned.mean:.4f} does not beat heuristic "
        f"{heuristic.mean:.4f} (a fine outcome — the heuristic is the product).",
    )
