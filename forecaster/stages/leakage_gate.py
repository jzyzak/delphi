"""Leakage-judge gate (C4.6).

Defense-in-depth over the structural PIT guarantee (CLAUDE.md §6): assemble the
forecast trace, run the high-recall judge over it, and quarantine on a flag. The
gate never silently drops a leak — a flagged forecast is quarantined with an
audit record for a human to disposition.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.forecast.ensemble import EnsembleForecast
from core.forecast.leakage_judge import (
    LeakageJudge,
    LeakageVerdict,
    QuarantineRecord,
    audit_and_quarantine,
    trace_from_ensemble,
    trace_from_supervisor,
)
from core.forecast.supervisor import ReconciledForecast

__all__ = ["LeakageGateResult", "run_leakage_gate"]


@dataclass(frozen=True)
class LeakageGateResult:
    """Outcome of the leakage gate over a forecast's traces."""

    flagged: bool
    verdicts: tuple[LeakageVerdict, ...]
    quarantine: tuple[QuarantineRecord, ...]


def run_leakage_gate(
    ensemble: EnsembleForecast,
    reconciled: ReconciledForecast,
    *,
    judge: LeakageJudge,
    forecast_id: str = "",
) -> LeakageGateResult:
    """Audit the ensemble and supervisor traces; quarantine any flagged trace."""
    traces = [
        trace_from_ensemble(ensemble, forecast_id=forecast_id),
        trace_from_supervisor(reconciled, forecast_id=forecast_id),
    ]
    verdicts: list[LeakageVerdict] = []
    quarantine: list[QuarantineRecord] = []
    for trace in traces:
        verdict, record = audit_and_quarantine(judge, trace)
        verdicts.append(verdict)
        if record is not None:
            quarantine.append(record)
    return LeakageGateResult(
        flagged=bool(quarantine),
        verdicts=tuple(verdicts),
        quarantine=tuple(quarantine),
    )
