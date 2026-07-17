"""Tests for the Tavily search provider adapter (hermetic; network mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from common.http.client import HttpClient
from common.secrets import EnvSecretProvider
from core.forecast.search import AsOfSearcher
from sources.providers.tavily import (
    TAVILY_API_KEY_SECRET,
    TavilySearchClient,
    tavily_config,
)
from sources.searcher import build_as_of_searcher
from sources.snapshot import InMemorySnapshotStore

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _secrets() -> EnvSecretProvider:
    return EnvSecretProvider({"DELPHI_SECRET_TAVILY_API_KEY": "tvly-secret"})


def _results() -> list[dict[str, Any]]:
    return [
        {"url": "http://before", "content": "old", "published_date": "2024-01-01", "score": 0.9},
        {"url": "http://after", "content": "leaked", "published_date": "2024-12-31"},
        {"url": "http://undated", "content": "no date", "published_date": None},
    ]


def test_config_defaults_target_tavily() -> None:
    cfg = tavily_config()
    assert cfg.base_url == "https://api.tavily.com/search"
    assert cfg.provider == "tavily"
    # v2 = server-side end_date bound; retires v1 (as-of-blind) snapshots.
    assert cfg.version == "v2"
    assert cfg.query_param == "query"
    assert cfg.api_key_secret == TAVILY_API_KEY_SECRET
    assert cfg.extra_params == {"search_depth": "advanced", "topic": "news"}


def test_search_posts_bearer_body_and_maps_results() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        import json as _json

        seen["body"] = _json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"results": _results()})

    client = TavilySearchClient(http=_http(handler), secrets=_secrets())
    response = client.search("will it happen", max_results=5)

    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["auth"] == "Bearer tvly-secret"
    assert seen["body"] == {
        "query": "will it happen",
        "max_results": 5,
        "search_depth": "advanced",
        "topic": "news",
    }
    assert [r.url for r in response.results] == ["http://before", "http://after", "http://undated"]
    assert response.raw == {"pages": [{"results": _results()}]}


def test_search_with_as_of_sends_server_side_end_date() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"results": []})

    client = TavilySearchClient(http=_http(handler), secrets=_secrets())
    client.search("q", as_of=datetime(2024, 6, 1, 12, 30, tzinfo=UTC))
    assert seen["body"]["end_date"] == "2024-06-01"


def test_search_as_of_bound_is_utc_date() -> None:
    from datetime import timedelta, timezone

    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"results": []})

    client = TavilySearchClient(http=_http(handler), secrets=_secrets())
    # 01:00 at UTC+3 is 22:00 the previous day in UTC — the bound must not
    # admit an extra local-time day past the ceiling.
    client.search("q", as_of=datetime(2024, 6, 1, 1, 0, tzinfo=timezone(timedelta(hours=3))))
    assert seen["body"]["end_date"] == "2024-05-31"


def test_search_without_as_of_omits_end_date() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"results": []})

    client = TavilySearchClient(http=_http(handler), secrets=_secrets())
    client.search("q")
    assert "end_date" not in seen["body"]


def test_search_without_secrets_sends_no_auth() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"results": []})

    client = TavilySearchClient(http=_http(handler))
    client.search("q")
    assert seen["auth"] is None


def test_search_tolerates_non_dict_payload() -> None:
    client = TavilySearchClient(http=_http(lambda _r: httpx.Response(200, json=["unexpected"])))
    assert client.search("q").results == ()


def test_search_tolerates_non_list_results() -> None:
    client = TavilySearchClient(
        http=_http(lambda _r: httpx.Response(200, json={"results": "nope"}))
    )
    assert client.search("q").results == ()


def test_search_rejects_bad_max_results() -> None:
    client = TavilySearchClient(http=_http(lambda _r: httpx.Response(200, json={})))
    try:
        client.search("q", max_results=0)
    except ValueError as exc:
        assert "max_results" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError")


def test_end_to_end_through_searcher_filters_and_snapshots() -> None:
    calls = {"n": 0}
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        calls["n"] += 1
        bodies.append(_json.loads(req.content.decode("utf-8")))
        return httpx.Response(200, json={"results": _results()})

    http = _http(handler)
    client = TavilySearchClient(http=http, secrets=_secrets())
    store = InMemorySnapshotStore()
    searcher = build_as_of_searcher(http_client=http, client=client, snapshot_store=store)

    assert isinstance(searcher, AsOfSearcher)
    evidence = searcher.as_of_search("q", as_of=AS_OF)
    # The searcher threads the ceiling into the provider request (§2.1).
    assert bodies[0]["end_date"] == "2024-06-01"
    # Only the pre-as-of, dated result survives; provider tag propagates.
    assert [e.source_id for e in evidence] == ["http://before"]
    assert evidence[0].source == "tavily"

    # Snapshot-first replay: no second network call for the same ceiling.
    again = searcher.as_of_search("q", as_of=AS_OF)
    assert again == evidence
    assert calls["n"] == 1
    assert len(store) == 1
