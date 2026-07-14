"""Proper-score reward + as-of-safe loop guard (A.3).

The single most important adaptation of the Conductor line of work for
forecasting (CLAUDE.md §4): the learned conductor is rewarded on **proper-score
improvement**, never on binary correctness — a binary-reward conductor learns
overconfidence. Lower proper score is better, so reward is ``baseline - candidate``
(positive when the candidate workflow scored better). The training loop is also
**as-of safe**: no example may reference evidence dated after its as-of time.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from core.pit.models import ensure_utc

__all__ = ["as_of_safe", "assert_as_of_safe", "proper_score_reward"]


def proper_score_reward(*, baseline: float, candidate: float) -> float:
    """Reward = improvement in proper score (lower is better), not correctness.

    Positive when ``candidate`` beats ``baseline`` (has a lower proper score).
    """
    return baseline - candidate


def as_of_safe(as_of: datetime, knowledge_times: Iterable[datetime]) -> bool:
    """True iff every evidence knowledge-time is at or before ``as_of`` (§2.1)."""
    ceiling = ensure_utc(as_of)
    return all(ensure_utc(kt) <= ceiling for kt in knowledge_times)


def assert_as_of_safe(as_of: datetime, knowledge_times: Iterable[datetime]) -> None:
    """Raise if any evidence would let the loop see past ``as_of`` (§2.1)."""
    ceiling = ensure_utc(as_of)
    for kt in knowledge_times:
        if ensure_utc(kt) > ceiling:
            msg = (
                f"as-of violation in training loop: evidence dated {ensure_utc(kt).isoformat()} "
                f"is later than as-of {ceiling.isoformat()}."
            )
            raise ValueError(msg)
