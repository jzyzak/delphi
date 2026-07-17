"""SourcesAsOfSearcher (C3.4): the app-layer ``AsOfSearcher`` implementation.

Composes the hosted client, the as-of filter, and the snapshot store into the
seam the forecast chain consumes (:class:`~core.forecast.search.AsOfSearcher`).
Reads are **snapshot-first**: a prior snapshot for ``(query, as_of)`` is replayed
before any network call, so retrospective runs are reproducible and never hit the
provider twice for the same ceiling. Every returned item is guaranteed
``knowledge_time <= as_of`` by the filter (Prime Directive §2.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import structlog

from common.http.client import HttpClient
from common.http.errors import HttpError
from common.secrets import SecretProvider
from core.forecast.search import AsOfSearcher, Evidence
from sources.asof_filter import filter_as_of
from sources.providers.hosted import HostedSearchClient, HostedSearchConfig
from sources.snapshot import (
    InMemorySnapshotStore,
    Snapshot,
    SnapshotStore,
    snapshot_key,
)

__all__ = [
    "CompositeAsOfSearcher",
    "SourcesAsOfSearcher",
    "build_as_of_searcher",
]

_LOG = structlog.get_logger(__name__)


class SourcesAsOfSearcher:
    """As-of search backed by a hosted provider + snapshot cache."""

    def __init__(
        self,
        *,
        client: HostedSearchClient,
        snapshot_store: SnapshotStore,
        max_results: int = 10,
    ) -> None:
        self._client = client
        self._snapshots = snapshot_store
        self._max_results = max_results

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        config = self._client.config
        key = snapshot_key(
            query=query, as_of=as_of, provider=config.provider, version=config.version
        )
        cached = self._snapshots.read(key)
        if cached is not None:
            return cached.evidence
        # The ceiling rides the provider request itself (server-side bound)
        # where supported; filter_as_of below stays the exact-timestamp gate.
        response = self._client.search(query, max_results=self._max_results, as_of=as_of)
        evidence = filter_as_of(
            response.results, as_of=as_of, provider=config.provider, query=query
        )
        self._snapshots.write(
            Snapshot(
                key=key,
                query=query,
                as_of=as_of,
                provider=config.provider,
                version=config.version,
                raw=response.raw,
                evidence=evidence,
            )
        )
        return evidence


class CompositeAsOfSearcher:
    """Fan-out over multiple ``AsOfSearcher`` providers, merged and ranked.

    Providers are queried sequentially; their evidence is concatenated,
    deduplicated by ``(source, source_id)`` keeping the highest-scored copy,
    sorted by score descending, and capped at ``max_items``. A provider raising
    :class:`~common.http.errors.HttpError` is logged and skipped — one provider
    outage must not kill evidence gathering. Each member enforces its own as-of
    ceiling, so the composite never widens the knowledge window.
    """

    def __init__(self, searchers: Sequence[AsOfSearcher], *, max_items: int = 10) -> None:
        if max_items < 1:
            msg = "max_items must be >= 1."
            raise ValueError(msg)
        self._searchers = tuple(searchers)
        self._max_items = max_items

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        best: dict[tuple[str, str], Evidence] = {}
        for searcher in self._searchers:
            try:
                items = searcher.as_of_search(query, as_of=as_of)
            except HttpError as exc:
                _LOG.warning(
                    "sources.composite.provider_failed",
                    provider=type(searcher).__name__,
                    query=query,
                    error=str(exc),
                )
                continue
            for item in items:
                key = (item.source, item.source_id)
                current = best.get(key)
                if current is None or item.score > current.score:
                    best[key] = item
        ranked = sorted(best.values(), key=lambda e: (-e.score, e.source, e.source_id))
        return tuple(ranked[: self._max_items])


def build_as_of_searcher(
    *,
    http_client: HttpClient,
    config: HostedSearchConfig | None = None,
    secret_provider: SecretProvider | None = None,
    snapshot_store: SnapshotStore | None = None,
    max_results: int = 10,
    client: HostedSearchClient | None = None,
) -> SourcesAsOfSearcher:
    """Wire a hosted client + snapshot store into a ``SourcesAsOfSearcher``.

    This is the composition entry point for the hosted profile; the ``test``
    profile uses :class:`~core.forecast.search.FixtureAsOfSearch` directly. A
    pre-built ``client`` (e.g. the Tavily adapter) may be injected; otherwise a
    generic :class:`HostedSearchClient` is built from ``config``.
    """
    resolved_client = client or HostedSearchClient(
        http=http_client, config=config or HostedSearchConfig(), secrets=secret_provider
    )
    # Explicit None check: an empty InMemorySnapshotStore is falsy (defines
    # __len__), so `store or default` would wrongly discard a caller's store.
    store = InMemorySnapshotStore() if snapshot_store is None else snapshot_store
    return SourcesAsOfSearcher(
        client=resolved_client,
        snapshot_store=store,
        max_results=max_results,
    )
