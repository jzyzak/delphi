"""Adapter conformance + snapshot-first + e2e leakage tests for the searcher (C3.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from common.composition import build_test_composition
from common.http.client import HttpClient
from core.forecast.search import AsOfSearcher
from sources.providers.hosted import HostedSearchConfig
from sources.searcher import SourcesAsOfSearcher, build_as_of_searcher
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


class TestCompositionWiring:
    def test_composition_builds_hosted_searcher(self) -> None:
        http, _ = _http_counting(_mixed_results())
        searcher = build_test_composition().hosted_searcher(http_client=http)
        assert isinstance(searcher, AsOfSearcher)
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["http://before"]
