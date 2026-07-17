"""SourcesAsOfSearcher (C3.4): the app-layer ``AsOfSearcher`` implementation.

Composes the hosted client, the as-of filter, and the snapshot store into the
seam the forecast chain consumes (:class:`~core.forecast.search.AsOfSearcher`).
Reads are **snapshot-first**: a prior snapshot for ``(query, as_of)`` is replayed
before any network call, so retrospective runs are reproducible and never hit the
provider twice for the same ceiling. Every returned item is guaranteed
``knowledge_time <= as_of`` by the filter (Prime Directive §2.1).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime

import structlog

from common.http.client import HttpClient
from common.http.errors import HttpError, HttpRateLimited
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
    "CircuitBreakerAsOfSearcher",
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


class CircuitBreakerAsOfSearcher:
    """Skips a rate-limited provider for a cooldown instead of hammering it.

    Keyless APIs like GDELT put a throttled IP into an *extended* cooldown:
    once 429s start, every further call fails, prolongs the ban, and still
    pays the politeness interval — a whole eval run can spend hours learning
    the same "no" (observed: seven straight hours of 429s). After
    ``failure_threshold`` consecutive trip errors the breaker opens: calls
    return no evidence immediately, without touching the provider. When
    ``cooldown_s`` has elapsed, one probe call is let through — success
    closes the circuit, another trip error re-opens it for a fresh cooldown.

    Only ``trip_on`` errors (rate limits by default) count toward the
    threshold; other failures propagate unchanged and neither trip nor heal
    the breaker, so an interleaved outage error cannot mask a throttle. Trip
    errors are re-raised while the circuit is closed — the composite's
    skip-and-log handling stays intact. Note the wrapped searcher is
    snapshot-first, so an open circuit also skips cache replays for the
    cooldown; acceptable, because eval queries are rarely repeated within
    one run.

    ``clock`` is injectable (monotonic seconds) so tests never sleep.
    """

    def __init__(
        self,
        inner: AsOfSearcher,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 900.0,
        trip_on: tuple[type[Exception], ...] = (HttpRateLimited,),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            msg = f"failure_threshold must be >= 1, got {failure_threshold!r}"
            raise ValueError(msg)
        if cooldown_s <= 0:
            msg = f"cooldown_s must be > 0, got {cooldown_s!r}"
            raise ValueError(msg)
        self._inner = inner
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._trip_on = trip_on
        self._clock = clock
        self._consecutive_failures = 0
        self._open_until: float | None = None

    @property
    def is_open(self) -> bool:
        """True while calls are being skipped (cooldown not yet elapsed)."""
        return self._open_until is not None and self._clock() < self._open_until

    def as_of_search(self, query: str, *, as_of: datetime) -> Sequence[Evidence]:
        if self._open_until is not None and self._clock() < self._open_until:
            return ()
        # Closed, or half-open (cooldown elapsed): let this call probe through.
        try:
            items = self._inner.as_of_search(query, as_of=as_of)
        except self._trip_on:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._open_until = self._clock() + self._cooldown_s
                _LOG.warning(
                    "sources.circuit_breaker.open",
                    provider=type(self._inner).__name__,
                    consecutive_failures=self._consecutive_failures,
                    cooldown_s=self._cooldown_s,
                )
            raise
        if self._open_until is not None:
            _LOG.info(
                "sources.circuit_breaker.closed",
                provider=type(self._inner).__name__,
            )
        self._open_until = None
        self._consecutive_failures = 0
        return items


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
