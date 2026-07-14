"""As-of knowledge-time filtering (C3.2) — the leakage firewall for retrieval.

Turns raw provider results into core :class:`~core.forecast.search.Evidence`,
enforcing Prime Directive §2.1: nothing dated after ``as_of`` survives, and an
*undated* result is treated as unsafe (dropped) because we cannot prove it is
older than the ceiling. ``as_of`` is always an explicit input; there is no
``now()`` here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from core.forecast.search import Evidence
from core.pit.models import ensure_utc
from sources.providers.hosted import HostedSearchResult

__all__ = [
    "filter_as_of",
    "parse_knowledge_time",
]


def parse_knowledge_time(raw: str | None) -> datetime | None:
    """Parse a published/updated date string into a tz-aware UTC datetime.

    Returns ``None`` for missing or unparseable values (which the filter treats
    as unsafe). Accepts ISO-8601 (including a trailing ``Z``) and date-only
    strings; a naive result is interpreted as UTC.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _to_evidence(
    result: HostedSearchResult, knowledge_time: datetime, *, provider: str, query: str
) -> Evidence:
    return Evidence(
        snippet=result.content or result.title,
        source=provider,
        source_id=result.url,
        knowledge_time=knowledge_time,
        score=result.score,
        query=query,
    )


def filter_as_of(
    results: Sequence[HostedSearchResult],
    *,
    as_of: datetime,
    provider: str,
    query: str = "",
) -> tuple[Evidence, ...]:
    """Drop leaked/undated results and map survivors to as-of ``Evidence``.

    A result survives iff it has a parseable ``published_date`` with
    ``knowledge_time <= as_of``. Output is deterministically ordered by
    ``(knowledge_time, source_id)``.
    """
    ceiling = ensure_utc(as_of)
    survivors: list[Evidence] = []
    for result in results:
        knowledge_time = parse_knowledge_time(result.published_date)
        if knowledge_time is None:
            continue  # undated: unsafe, cannot prove it predates the ceiling
        if knowledge_time > ceiling:
            continue  # leaked: dated after the as-of ceiling
        survivors.append(_to_evidence(result, knowledge_time, provider=provider, query=query))
    survivors.sort(key=lambda e: (e.knowledge_time, e.source_id))
    return tuple(survivors)
