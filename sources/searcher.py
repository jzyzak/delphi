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

from common.http.client import HttpClient
from common.secrets import SecretProvider
from core.forecast.search import Evidence
from sources.asof_filter import filter_as_of
from sources.providers.hosted import HostedSearchClient, HostedSearchConfig
from sources.snapshot import (
    InMemorySnapshotStore,
    Snapshot,
    SnapshotStore,
    snapshot_key,
)

__all__ = [
    "SourcesAsOfSearcher",
    "build_as_of_searcher",
]


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
        response = self._client.search(query, max_results=self._max_results)
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
