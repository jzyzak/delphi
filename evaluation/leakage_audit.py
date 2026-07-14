"""Leakage audit report (C6.7).

Before trusting any retrospective number, run the leakage judge over the suite
and report the leakage rate and the worst-case (flagged-at-chance) robustness
(CLAUDE.md §2.6). A great score on a leaky benchmark is noise; this quantifies
how much of the suite is trustworthy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.forecast.leakage_judge import (
    LeakageJudge,
    LeakageReport,
    RegistrySlice,
    Trace,
)

__all__ = ["LeakageAudit", "audit_suite"]


@dataclass(frozen=True)
class LeakageAudit:
    """Suite-level leakage audit: rate + worst-case robustness."""

    report: LeakageReport
    leakage_rate: float
    clean_fraction: float

    def render(self) -> str:
        lines = [
            f"leakage_rate={self.leakage_rate:.4f}",
            f"clean_fraction (flagged-at-chance robustness)={self.clean_fraction:.4f}",
            f"flagged={self.report.flagged}/{self.report.total}",
        ]
        for comp in self.report.by_component:
            lines.append(f"  {comp.component.value}: {comp.flagged}/{comp.total} ({comp.rate:.3f})")
        return "\n".join(lines)


def audit_suite(
    judge: LeakageJudge,
    traces: Sequence[Trace],
    *,
    slice_id: str = "",
    baseline: LeakageReport | None = None,
) -> LeakageAudit:
    """Batch-audit ``traces``; return the leakage rate and clean fraction."""
    report = judge.estimate_leakage_rate(
        RegistrySlice(traces=tuple(traces), slice_id=slice_id), baseline=baseline
    )
    rate = report.aggregate_rate
    return LeakageAudit(report=report, leakage_rate=rate, clean_fraction=1.0 - rate)
