"""GDELT DOC 2.0 historical evidence provider (free, keyless).

GDELT's DOC API can be bounded **at query time** with ``enddatetime``, which
makes it a true historical provider: a retrospective ``as_of`` yields articles
seen at or before that ceiling instead of an empty (or leaky) result set. The
provider implements the core :class:`~core.forecast.search.AsOfSearcher`
Protocol directly, with the same snapshot-first semantics as
:class:`~sources.searcher.SourcesAsOfSearcher`, and still runs every result
through :func:`~sources.asof_filter.filter_as_of` as belt-and-suspenders
(Prime Directive §2.1) — the API bound is not trusted on its own.

API shape assumed (documented, keyless): ``GET
https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=artlist&format=json``
returns ``{"articles": [{"url", "title", "seendate", "domain", ...}]}`` where
``seendate`` is ``YYYYMMDDTHHMMSSZ`` (UTC). GDELT provides no snippet in
artlist mode, so the article title doubles as the evidence snippet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from common.http.client import HttpClient
from core.forecast.search import Evidence
from core.pit.models import ensure_utc
from sources.asof_filter import filter_as_of
from sources.providers.hosted import HostedSearchResponse, HostedSearchResult
from sources.snapshot import (
    InMemorySnapshotStore,
    Snapshot,
    SnapshotStore,
    snapshot_key,
)

__all__ = [
    "GdeltAsOfSearcher",
    "GdeltConfig",
]

_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_DATETIME_FORMAT = "%Y%m%d%H%M%S"
_SEENDATE_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class GdeltConfig:
    """Endpoint + query-shaping knobs for the GDELT DOC 2.0 API."""

    base_url: str = _GDELT_DOC_URL
    provider: str = "gdelt"
    version: str = "v2"
    sort: str = "hybridrel"
    # GDELT expects an explicit window; without a start it defaults to a recent
    # slice, which is useless for retrospective ceilings. The window ends at
    # ``as_of`` and reaches back this many days.
    lookback_days: int = 90


def parse_seendate(raw: Any) -> str | None:
    """Parse GDELT's ``YYYYMMDDTHHMMSSZ`` seendate into an ISO-8601 UTC string.

    Returns ``None`` for missing or malformed values so the as-of filter treats
    the article as undated (unsafe → dropped).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.strptime(raw.strip(), _SEENDATE_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed.isoformat()


class GdeltAsOfSearcher:
    """As-of evidence from GDELT DOC 2.0 (implements ``AsOfSearcher``).

    Reads are snapshot-first: a prior snapshot for ``(query, as_of)`` replays
    before any network call. On a miss, the HTTP query itself is bounded at the
    ceiling (``enddatetime=as_of``) and the mapped results are additionally
    passed through :func:`filter_as_of`, so nothing dated after ``as_of`` can
    survive even if the API is sloppy about its own bound.
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        config: GdeltConfig | None = None,
        snapshot_store: SnapshotStore | None = None,
        max_results: int = 10,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1."
            raise ValueError(msg)
        self._http = http
        self._config = config or GdeltConfig()
        # Explicit None check: an empty InMemorySnapshotStore is falsy.
        self._snapshots = InMemorySnapshotStore() if snapshot_store is None else snapshot_store
        self._max_results = max_results

    @property
    def config(self) -> GdeltConfig:
        return self._config

    def search(self, query: str, *, max_results: int, as_of: datetime) -> HostedSearchResponse:
        """Query GDELT bounded at ``as_of`` and map articles to typed results.

        Unlike the hosted providers, ``as_of`` is a query parameter here
        (``enddatetime``) — the bound is applied at query time, not only in
        post-filtering.
        """
        if max_results < 1:
            msg = "max_results must be >= 1."
            raise ValueError(msg)
        ceiling = ensure_utc(as_of)
        start = ceiling - timedelta(days=self._config.lookback_days)
        params: dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_results,
            "startdatetime": start.strftime(_GDELT_DATETIME_FORMAT),
            "enddatetime": ceiling.strftime(_GDELT_DATETIME_FORMAT),
            "sort": self._config.sort,
        }
        payload = self._http.get_json(self._config.base_url, params=params)
        raw = payload if isinstance(payload, dict) else {"articles": []}
        raw_articles = raw.get("articles", [])
        articles = raw_articles if isinstance(raw_articles, list) else []
        results = _map_articles(articles, max_results=max_results)
        return HostedSearchResponse(query=query, results=results, raw={"pages": [raw]})

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        """Snapshot-first as-of search; every survivor has knowledge_time <= as_of."""
        ceiling = ensure_utc(as_of)
        config = self._config
        key = snapshot_key(
            query=query, as_of=ceiling, provider=config.provider, version=config.version
        )
        cached = self._snapshots.read(key)
        if cached is not None:
            return cached.evidence
        response = self.search(query, max_results=self._max_results, as_of=ceiling)
        evidence = filter_as_of(
            response.results, as_of=ceiling, provider=config.provider, query=query
        )
        self._snapshots.write(
            Snapshot(
                key=key,
                query=query,
                as_of=ceiling,
                provider=config.provider,
                version=config.version,
                raw=response.raw,
                evidence=evidence,
            )
        )
        return evidence


def _map_articles(articles: list[Any], *, max_results: int) -> tuple[HostedSearchResult, ...]:
    """Map GDELT article objects onto the shared result contract, rank-scored."""
    sliced = [a for a in articles[:max_results] if isinstance(a, dict)]
    total = len(sliced)
    results: list[HostedSearchResult] = []
    for rank, article in enumerate(sliced):
        title = str(article.get("title", ""))
        results.append(
            HostedSearchResult(
                title=title,
                url=str(article.get("url", "")),
                # GDELT artlist mode returns no snippet; the title is the snippet.
                content=title,
                published_date=parse_seendate(article.get("seendate")),
                score=(total - rank) / total,
            )
        )
    return tuple(results)
