"""Adapter conformance + snapshot-first + e2e leakage tests for the searcher (C3.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from common.composition import build_test_composition
from common.http.client import HttpClient
from common.http.errors import HttpNotFound, HttpRateLimited
from core.forecast.search import AsOfSearcher, Evidence, FixtureAsOfSearch
from sources.providers.hosted import HostedSearchConfig
from sources.searcher import (
    CircuitBreakerAsOfSearcher,
    CompositeAsOfSearcher,
    SourcesAsOfSearcher,
    build_as_of_searcher,
)
from sources.snapshot import InMemorySnapshotStore

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _http_counting(results: list[dict[str, Any]]) -> tuple[HttpClient, dict[str, int]]:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"results": results})

    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler))), calls


def _mixed_results() -> list[dict[str, Any]]:
    return [
        {"url": "http://before", "content": "old", "published_date": "2024-01-01"},
        {"url": "http://after", "content": "leaked", "published_date": "2024-12-31"},
        {"url": "http://undated", "content": "no date", "published_date": None},
    ]


class TestSourcesAsOfSearcher:
    def test_is_asof_searcher(self) -> None:
        http, _ = _http_counting([])
        searcher = build_as_of_searcher(http_client=http)
        assert isinstance(searcher, AsOfSearcher)

    def test_e2e_leakage_only_past_evidence(self) -> None:
        http, _ = _http_counting(_mixed_results())
        searcher = build_as_of_searcher(http_client=http)
        evidence = searcher.as_of_search("will it happen", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["http://before"]
        assert all(e.knowledge_time <= AS_OF for e in evidence)

    def test_snapshot_first_replays_without_second_network_call(self) -> None:
        http, calls = _http_counting(_mixed_results())
        store = InMemorySnapshotStore()
        searcher = build_as_of_searcher(http_client=http, snapshot_store=store)
        first = searcher.as_of_search("q", as_of=AS_OF)
        second = searcher.as_of_search("q", as_of=AS_OF)
        assert first == second
        assert calls["n"] == 1  # second call served from the snapshot
        assert len(store) == 1

    def test_distinct_as_of_triggers_new_fetch(self) -> None:
        http, calls = _http_counting(_mixed_results())
        searcher = build_as_of_searcher(http_client=http, snapshot_store=InMemorySnapshotStore())
        searcher.as_of_search("q", as_of=AS_OF)
        searcher.as_of_search("q", as_of=datetime(2025, 1, 1, tzinfo=UTC))
        assert calls["n"] == 2

    def test_custom_config_provider_recorded_on_evidence(self) -> None:
        http, _ = _http_counting(_mixed_results())
        searcher = build_as_of_searcher(
            http_client=http, config=HostedSearchConfig(provider="tavily")
        )
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert evidence[0].source == "tavily"

    def test_direct_construction(self) -> None:
        http, _ = _http_counting([])
        searcher = build_as_of_searcher(http_client=http)
        assert isinstance(searcher, SourcesAsOfSearcher)


def _evidence(source: str, source_id: str, *, score: float) -> Evidence:
    return Evidence(
        snippet=f"snippet {source_id}",
        source=source,
        source_id=source_id,
        knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
        score=score,
        query="q",
    )


class _RaisingSearcher:
    """Fake provider whose as_of_search raises a given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.call_count = 0

    def as_of_search(self, query: str, *, as_of: datetime) -> tuple[Evidence, ...]:
        self.call_count += 1
        raise self._exc


class TestCompositionWiring:
    def test_composition_builds_hosted_searcher(self) -> None:
        http, _ = _http_counting(_mixed_results())
        searcher = build_test_composition().hosted_searcher(http_client=http)
        assert isinstance(searcher, AsOfSearcher)
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["http://before"]


class TestCompositeAsOfSearcher:
    def test_is_asof_searcher(self) -> None:
        assert isinstance(CompositeAsOfSearcher([]), AsOfSearcher)

    def test_merges_across_providers_sorted_by_score_desc(self) -> None:
        a = FixtureAsOfSearch(default=[_evidence("gdelt", "http://a", score=0.4)])
        b = FixtureAsOfSearch(default=[_evidence("wikipedia", "wikipedia:1:2", score=0.9)])
        composite = CompositeAsOfSearcher([a, b])

        evidence = composite.as_of_search("q", as_of=AS_OF)

        assert [e.source_id for e in evidence] == ["wikipedia:1:2", "http://a"]
        assert [e.score for e in evidence] == [0.9, 0.4]
        assert a.call_count == 1
        assert b.call_count == 1

    def test_passes_query_and_as_of_through(self) -> None:
        inner = FixtureAsOfSearch()
        CompositeAsOfSearcher([inner]).as_of_search("will it happen", as_of=AS_OF)
        assert inner.queries == [("will it happen", AS_OF)]

    def test_dedupes_by_source_and_source_id_keeping_highest_score(self) -> None:
        low = FixtureAsOfSearch(default=[_evidence("gdelt", "http://dup", score=0.2)])
        high = FixtureAsOfSearch(default=[_evidence("gdelt", "http://dup", score=0.8)])
        evidence = CompositeAsOfSearcher([low, high]).as_of_search("q", as_of=AS_OF)
        assert len(evidence) == 1
        assert evidence[0].score == 0.8

    def test_same_id_different_source_not_deduped(self) -> None:
        a = FixtureAsOfSearch(default=[_evidence("gdelt", "shared-id", score=0.5)])
        b = FixtureAsOfSearch(default=[_evidence("wikipedia", "shared-id", score=0.5)])
        evidence = CompositeAsOfSearcher([a, b]).as_of_search("q", as_of=AS_OF)
        assert {e.source for e in evidence} == {"gdelt", "wikipedia"}

    def test_caps_at_max_items(self) -> None:
        items = [_evidence("gdelt", f"http://x/{i}", score=(10 - i) / 10) for i in range(6)]
        composite = CompositeAsOfSearcher([FixtureAsOfSearch(default=items)], max_items=3)
        evidence = composite.as_of_search("q", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["http://x/0", "http://x/1", "http://x/2"]

    def test_default_cap_is_ten(self) -> None:
        items = [_evidence("gdelt", f"http://x/{i:02d}", score=1.0) for i in range(15)]
        evidence = CompositeAsOfSearcher([FixtureAsOfSearch(default=items)]).as_of_search(
            "q", as_of=AS_OF
        )
        assert len(evidence) == 10

    def test_http_error_provider_is_skipped_and_others_still_return(self) -> None:
        boom = _RaisingSearcher(HttpNotFound("provider down"))
        healthy = FixtureAsOfSearch(default=[_evidence("wikipedia", "wikipedia:1:2", score=0.7)])
        evidence = CompositeAsOfSearcher([boom, healthy]).as_of_search("q", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["wikipedia:1:2"]
        assert boom.call_count == 1
        assert healthy.call_count == 1

    def test_non_http_error_propagates(self) -> None:
        boom = _RaisingSearcher(ValueError("logic bug"))
        with pytest.raises(ValueError, match="logic bug"):
            CompositeAsOfSearcher([boom]).as_of_search("q", as_of=AS_OF)

    def test_failing_provider_last_still_returns_earlier_results(self) -> None:
        healthy = FixtureAsOfSearch(default=[_evidence("gdelt", "http://ok", score=0.6)])
        boom = _RaisingSearcher(HttpNotFound("provider down"))
        evidence = CompositeAsOfSearcher([healthy, boom]).as_of_search("q", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["http://ok"]

    def test_score_ties_break_deterministically(self) -> None:
        a = FixtureAsOfSearch(default=[_evidence("wikipedia", "b", score=0.5)])
        b = FixtureAsOfSearch(default=[_evidence("gdelt", "a", score=0.5)])
        evidence = CompositeAsOfSearcher([a, b]).as_of_search("q", as_of=AS_OF)
        assert [(e.source, e.source_id) for e in evidence] == [("gdelt", "a"), ("wikipedia", "b")]

    def test_empty_provider_list_returns_empty(self) -> None:
        assert CompositeAsOfSearcher([]).as_of_search("q", as_of=AS_OF) == ()

    def test_rejects_bad_max_items(self) -> None:
        with pytest.raises(ValueError, match="max_items"):
            CompositeAsOfSearcher([], max_items=0)


class _FakeClock:
    """Deterministic monotonic clock: tests advance it explicitly, never sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FlakySearcher:
    """Inner searcher whose per-call behavior is scripted by the test."""

    def __init__(self) -> None:
        self.exc: Exception | None = HttpRateLimited("429")
        self.call_count = 0

    def as_of_search(self, query: str, *, as_of: datetime) -> tuple[Evidence, ...]:
        self.call_count += 1
        if self.exc is not None:
            raise self.exc
        return (_evidence("gdelt", "http://x", score=0.5),)


class TestCircuitBreakerAsOfSearcher:
    def _breaker(
        self, inner: _FlakySearcher, clock: _FakeClock, *, threshold: int = 3
    ) -> CircuitBreakerAsOfSearcher:
        return CircuitBreakerAsOfSearcher(
            inner, failure_threshold=threshold, cooldown_s=900.0, clock=clock
        )

    def _trip(self, breaker: CircuitBreakerAsOfSearcher, times: int) -> None:
        for _ in range(times):
            with pytest.raises(HttpRateLimited):
                breaker.as_of_search("q", as_of=AS_OF)

    def test_is_asof_searcher(self) -> None:
        breaker = CircuitBreakerAsOfSearcher(FixtureAsOfSearch(), clock=_FakeClock())
        assert isinstance(breaker, AsOfSearcher)

    def test_below_threshold_failures_propagate_and_keep_calling(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 2)
        assert breaker.is_open is False
        assert inner.call_count == 2  # still closed: every call reaches the provider

    def test_opens_after_threshold_and_skips_without_calling_inner(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 3)
        assert breaker.is_open is True
        assert breaker.as_of_search("q", as_of=AS_OF) == ()
        assert breaker.as_of_search("q2", as_of=AS_OF) == ()
        assert inner.call_count == 3  # the open circuit never touched the provider

    def test_success_resets_the_consecutive_count(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 2)
        inner.exc = None
        assert len(breaker.as_of_search("q", as_of=AS_OF)) == 1
        inner.exc = HttpRateLimited("429")
        self._trip(breaker, 2)  # a fresh streak: 2 < threshold, still closed
        assert breaker.is_open is False

    def test_probe_after_cooldown_success_closes(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 3)
        clock.advance(900.0)
        inner.exc = None
        assert len(breaker.as_of_search("q", as_of=AS_OF)) == 1  # the probe went through
        assert breaker.is_open is False
        assert len(breaker.as_of_search("q2", as_of=AS_OF)) == 1
        assert inner.call_count == 5

    def test_probe_failure_reopens_for_a_fresh_cooldown(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 3)
        clock.advance(900.0)
        self._trip(breaker, 1)  # the probe itself rate-limits
        assert breaker.is_open is True
        clock.advance(899.0)
        assert breaker.as_of_search("q", as_of=AS_OF) == ()  # still cooling down
        assert inner.call_count == 4

    def test_non_trip_errors_propagate_without_counting_or_healing(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 2)
        inner.exc = HttpNotFound("404")
        with pytest.raises(HttpNotFound):
            breaker.as_of_search("q", as_of=AS_OF)
        assert breaker.is_open is False
        inner.exc = HttpRateLimited("429")
        self._trip(breaker, 1)  # 404 did not reset the streak: 3rd rate limit opens
        assert breaker.is_open is True

    def test_composite_skips_open_breaker_silently(self) -> None:
        inner, clock = _FlakySearcher(), _FakeClock()
        breaker = self._breaker(inner, clock)
        self._trip(breaker, 3)
        other = FixtureAsOfSearch(default=[_evidence("wikipedia", "http://w", score=0.9)])
        evidence = CompositeAsOfSearcher([breaker, other]).as_of_search("q", as_of=AS_OF)
        assert [e.source for e in evidence] == ["wikipedia"]
        assert inner.call_count == 3

    def test_rejects_bad_threshold_and_cooldown(self) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerAsOfSearcher(FixtureAsOfSearch(), failure_threshold=0)
        with pytest.raises(ValueError, match="cooldown_s"):
            CircuitBreakerAsOfSearcher(FixtureAsOfSearch(), cooldown_s=0)
