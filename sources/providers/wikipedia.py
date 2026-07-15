"""Wikipedia revision-history evidence provider (free, keyless, truly historical).

MediaWiki keeps every revision of every page, so the article text **as it stood
at any past moment** is retrievable: search titles, then fetch the newest
revision at or before ``as_of`` (``rvstart=<as_of>&rvdir=older&rvlimit=1``).
The revision timestamp is the knowledge time and is guaranteed ``<= as_of`` by
the API's own ordering — and the results are still run through
:func:`~sources.asof_filter.filter_as_of` as belt-and-suspenders (Prime
Directive §2.1). Implements the core
:class:`~core.forecast.search.AsOfSearcher` Protocol with the same
snapshot-first semantics as :class:`~sources.searcher.SourcesAsOfSearcher`.

API shape assumed (``formatversion=2`` JSON):

- ``action=query&list=search`` → ``{"query": {"search": [{"pageid", "title"}]}}``
- ``action=query&prop=revisions&rvprop=ids|timestamp|content&rvslots=main`` →
  ``{"query": {"pages": [{"pageid", "title", "revisions": [{"revid",
  "timestamp" (ISO-8601 Z), "slots": {"main": {"content": "..."}}}]}]}}``;
  a page with no revision at/before ``as_of`` has no ``revisions`` key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from common.http.client import HttpClient
from core.forecast.search import Evidence
from core.pit.models import ensure_utc
from sources.asof_filter import filter_as_of
from sources.providers.hosted import HostedSearchResult
from sources.snapshot import (
    InMemorySnapshotStore,
    Snapshot,
    SnapshotStore,
    snapshot_key,
)

__all__ = [
    "WikipediaAsOfSearcher",
    "WikipediaConfig",
]

_WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
_MEDIAWIKI_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class WikipediaConfig:
    """Endpoint + snippet knobs for the MediaWiki API."""

    base_url: str = _WIKIPEDIA_API_URL
    provider: str = "wikipedia"
    version: str = "v1"
    # Revision wikitext is long; truncate to a snippet-sized lead.
    snippet_chars: int = 600


class WikipediaAsOfSearcher:
    """As-of evidence from Wikipedia revision history (implements ``AsOfSearcher``).

    Two-call flow per query: one title search, then one revision lookup per
    title pinned at the ceiling. ``source_id`` is
    ``wikipedia:<pageid>:<revid>``, a fully reproducible pointer to the exact
    revision used.
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        config: WikipediaConfig | None = None,
        snapshot_store: SnapshotStore | None = None,
        max_results: int = 5,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1."
            raise ValueError(msg)
        self._http = http
        self._config = config or WikipediaConfig()
        # Explicit None check: an empty InMemorySnapshotStore is falsy.
        self._snapshots = InMemorySnapshotStore() if snapshot_store is None else snapshot_store
        self._max_results = max_results

    @property
    def config(self) -> WikipediaConfig:
        return self._config

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
        results, raw = self._fetch(query, as_of=ceiling)
        evidence = filter_as_of(results, as_of=ceiling, provider=config.provider, query=query)
        self._snapshots.write(
            Snapshot(
                key=key,
                query=query,
                as_of=ceiling,
                provider=config.provider,
                version=config.version,
                raw=raw,
                evidence=evidence,
            )
        )
        return evidence

    def _fetch(
        self, query: str, *, as_of: datetime
    ) -> tuple[tuple[HostedSearchResult, ...], dict[str, Any]]:
        """Search titles, then pin each page at its newest revision <= ``as_of``."""
        search_payload = self._http.get_json(
            self._config.base_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": self._max_results,
                "format": "json",
                "formatversion": 2,
            },
        )
        titles = _extract_titles(search_payload)
        total = len(titles)
        results: list[HostedSearchResult] = []
        revision_payloads: list[Any] = []
        for rank, title in enumerate(titles):
            revision_payload = self._http.get_json(
                self._config.base_url,
                params={
                    "action": "query",
                    "prop": "revisions",
                    "titles": title,
                    "rvlimit": 1,
                    "rvdir": "older",
                    "rvstart": as_of.strftime(_MEDIAWIKI_TIMESTAMP_FORMAT),
                    "rvprop": "ids|timestamp|content",
                    "rvslots": "main",
                    "format": "json",
                    "formatversion": 2,
                },
            )
            revision_payloads.append(revision_payload)
            result = _revision_to_result(
                revision_payload,
                title=title,
                score=(total - rank) / total,
                snippet_chars=self._config.snippet_chars,
            )
            if result is not None:
                results.append(result)
        raw = {
            "pages": [
                {"search": search_payload, "revisions": revision_payloads},
            ]
        }
        return tuple(results), raw


def _extract_titles(payload: Any) -> list[str]:
    """Pull search-hit titles out of a ``list=search`` response, tolerantly."""
    if not isinstance(payload, dict):
        return []
    query = payload.get("query")
    if not isinstance(query, dict):
        return []
    hits = query.get("search")
    if not isinstance(hits, list):
        return []
    titles: list[str] = []
    for hit in hits:
        if isinstance(hit, dict) and str(hit.get("title", "")):
            titles.append(str(hit["title"]))
    return titles


def _revision_to_result(
    payload: Any, *, title: str, score: float, snippet_chars: int
) -> HostedSearchResult | None:
    """Map a revision lookup onto the shared result contract.

    Returns ``None`` when the page has no revision at/before the ceiling (the
    page did not exist yet) or the payload is malformed — the page is skipped.
    """
    page = _first_page(payload)
    if page is None:
        return None
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        return None  # no revision at/before as_of: the page is unknowable then
    revision = revisions[0]
    if not isinstance(revision, dict):
        return None
    timestamp = revision.get("timestamp")
    revid = revision.get("revid")
    pageid = page.get("pageid")
    if not isinstance(timestamp, str) or revid is None or pageid is None:
        return None
    content = _revision_content(revision)
    return HostedSearchResult(
        title=title,
        # source_id: reproducible pointer to the exact revision used.
        url=f"wikipedia:{pageid}:{revid}",
        content=content[:snippet_chars] or title,
        published_date=timestamp,
        score=score,
    )


def _first_page(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    query = payload.get("query")
    if not isinstance(query, dict):
        return None
    pages = query.get("pages")
    if not isinstance(pages, list) or not pages:
        return None
    page = pages[0]
    return page if isinstance(page, dict) else None


def _revision_content(revision: dict[str, Any]) -> str:
    slots = revision.get("slots")
    if not isinstance(slots, dict):
        return ""
    main = slots.get("main")
    if not isinstance(main, dict):
        return ""
    return str(main.get("content", ""))
