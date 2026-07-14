"""Live benchmark loop (Phase 9).

The nightly loop that produces the only real number (CLAUDE.md §2.7): harvest
genuinely-open questions, forecast them via the conductor, and — once they close
— resolve and score them. The live number is untunable by construction because
the answers do not exist at forecast time. Runs are made idempotent through the
run-state claim, so a re-run of the same nightly tick is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from core.orchestration.run_state import RunStateStore
from core.orchestration.schedules import Cadence
from core.orchestration.types import ClaimResult, LoopName

__all__ = ["ClaimOutcome", "claim_and_run", "live_cadence"]


def live_cadence() -> Cadence:
    """Nightly cadence for the live loop (C9.3.a)."""
    return Cadence(interval=timedelta(days=1))


class ClaimOutcome:
    """String constants describing a claimed-run outcome (for CLI rendering)."""

    RAN = "ran"
    SKIPPED = "already_succeeded"


def claim_and_run[T](
    run_state: RunStateStore,
    *,
    step_id: str,
    tick_at: datetime,
    action: Callable[[], T],
    loop_name: LoopName = LoopName.MONITORING,
) -> tuple[str, T | None]:
    """Claim ``step_id`` and run ``action`` once; skip if already succeeded (C9.3.c).

    Returns ``(outcome, result)``: on a fresh claim the action runs and its result
    is returned; on an already-succeeded step the action is not re-run. A failure
    is recorded to the run-state store (so a restart may retry) and re-raised.
    """
    claim = run_state.claim_step(step_id=step_id, loop_name=loop_name, tick_at=tick_at)
    if claim is ClaimResult.ALREADY_SUCCEEDED:
        return ClaimOutcome.SKIPPED, None
    try:
        result = action()
    except Exception as exc:
        run_state.mark_failed(step_id=step_id, error_message=str(exc))
        raise
    run_state.mark_succeeded(step_id=step_id)
    return ClaimOutcome.RAN, result
