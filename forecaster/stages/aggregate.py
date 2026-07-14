"""Aggregate + supervisor reconcile stage (C4.4).

The robust ensemble already produced a log-odds-friendly aggregate + spread. This
stage runs the disciplined :class:`~core.forecast.supervisor.Supervisor`, which
detects material disagreement, resolves it with a *targeted as-of search*, and
applies the update ONLY at high confidence — otherwise falling back to the robust
aggregate unchanged (confidence gating, C4.4.d). It can improve on or equal the
aggregate, never underperform it.
"""

from __future__ import annotations

from core.forecast.ensemble import EnsembleForecast
from core.forecast.search import AsOfSearcher
from core.forecast.supervisor import (
    InMemoryReconciliationCache,
    ReconciledForecast,
    ReconciliationCache,
    Supervisor,
    SupervisorLLM,
    detect_disagreement,
)

__all__ = ["detect_disagreement", "reconcile"]


def reconcile(
    ensemble: EnsembleForecast,
    *,
    searcher: AsOfSearcher,
    supervisor_llm: SupervisorLLM,
    cache: ReconciliationCache | None = None,
) -> ReconciledForecast:
    """Reconcile ensemble disagreement via the supervisor's confidence-gated search."""
    supervisor = Supervisor(
        searcher,
        supervisor_llm,
        cache if cache is not None else InMemoryReconciliationCache(),
    )
    return supervisor.reconcile(ensemble)
