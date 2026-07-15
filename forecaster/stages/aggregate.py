"""Aggregate + supervisor reconcile stage (C4.4).

The robust ensemble already produced a log-odds-friendly aggregate + spread. This
stage runs the disciplined :class:`~core.forecast.supervisor.Supervisor`, which
detects material disagreement, resolves it with a *targeted as-of search*, and
applies the update ONLY at high confidence — otherwise falling back to the robust
aggregate unchanged (confidence gating, C4.4.d). It can improve on or equal the
aggregate, never underperform it.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.forecast.ensemble import EnsembleForecast
from core.forecast.search import AsOfSearcher
from core.forecast.supervisor import (
    DEFAULT_MULTIMODAL_GAP,
    DEFAULT_OUTLIER_STD_MULTIPLIER,
    DEFAULT_SPREAD_THRESHOLD,
    Confidence,
    InMemoryReconciliationCache,
    ReconciledForecast,
    ReconciliationCache,
    Supervisor,
    SupervisorLLM,
    detect_disagreement,
)

__all__ = ["SupervisorTuning", "detect_disagreement", "reconcile"]


@dataclass(frozen=True)
class SupervisorTuning:
    """Trigger/apply tuning for the reconciliation supervisor.

    Defaults mirror the supervisor's own conservative defaults; compositions
    that run larger ensembles may loosen the trigger and accept MEDIUM-
    confidence updates.
    """

    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD
    outlier_std_multiplier: float = DEFAULT_OUTLIER_STD_MULTIPLIER
    multimodal_gap: float = DEFAULT_MULTIMODAL_GAP
    min_apply_confidence: Confidence = Confidence.HIGH


def reconcile(
    ensemble: EnsembleForecast,
    *,
    searcher: AsOfSearcher,
    supervisor_llm: SupervisorLLM,
    cache: ReconciliationCache | None = None,
    tuning: SupervisorTuning | None = None,
) -> ReconciledForecast:
    """Reconcile ensemble disagreement via the supervisor's confidence-gated search."""
    tuning = tuning if tuning is not None else SupervisorTuning()
    supervisor = Supervisor(
        searcher,
        supervisor_llm,
        cache if cache is not None else InMemoryReconciliationCache(),
        spread_threshold=tuning.spread_threshold,
        outlier_std_multiplier=tuning.outlier_std_multiplier,
        multimodal_gap=tuning.multimodal_gap,
        min_apply_confidence=tuning.min_apply_confidence,
    )
    return supervisor.reconcile(ensemble)
