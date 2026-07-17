"""Unit tests for the hosted search client (C3.1). Transport-mocked, no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from common.http.client import HttpClient
from common.http.errors import HttpError
from common.secrets import EnvSecretProvider
from sources.providers.hosted import (
    HostedSearchClient,
    HostedSearchConfig,
    HostedSearchResult,
    as_of_date_bound,
)


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _result(url: str, date: str | None = "2024-01-01") -> dict[str, Any]:
    return {
        "title": f"t-{url}",
        "url": url,
        "content": "body",
        "published_date": date,
        "score": 0.5,
    }


class TestHostedSearchResultFromRaw:
    def test_full(self) -> None:
        r = HostedSearchResult.from_raw(_result("http://a"))
        assert r.url == "http://a"
        assert r.published_date == "2024-01-01"
        assert r.score == pytest.approx(0.5)

    def test_missing_fields_default(self) -> None:
        r = HostedSearchResult.from_raw({})
        assert r.title == "" and r.url == "" and r.published_date is None and r.score == 0.0

    def test_non_dict_returns_empty(self) -> None:
        assert HostedSearchResult.from_raw("nope").url == ""

    def test_bad_score_becomes_zero(self) -> None:
        assert HostedSearchResult.from_raw({"score": "high"}).score == 0.0

    def test_null_published_becomes_none(self) -> None:
        assert HostedSearchResult.from_raw({"published_date": None}).published_date is None


class TestAsOfDateBound:
    def test_renders_utc_date(self) -> None:
        assert as_of_date_bound(datetime(2024, 6, 1, 12, 30, tzinfo=UTC)) == "2024-06-01"

    def test_converts_non_utc_zone(self) -> None:
        bound = as_of_date_bound(datetime(2024, 6, 1, 1, 0, tzinfo=timezone(timedelta(hours=3))))
        assert bound == "2024-05-31"


class TestSearch:
    def test_success_maps_results(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [_result("http://a"), _result("http://b")]})

        client = HostedSearchClient(http=_http(handler))
        resp = client.search("q", max_results=5)
        assert [r.url for r in resp.results] == ["http://a", "http://b"]
        assert resp.raw["pages"]  # raw retained for snapshotting

    def test_empty_results(self) -> None:
        client = HostedSearchClient(
            http=_http(lambda _r: httpx.Response(200, json={"results": []}))
        )
        resp = client.search("q")
        assert resp.results == ()

    def test_non_dict_payload_yields_empty(self) -> None:
        client = HostedSearchClient(http=_http(lambda _r: httpx.Response(200, json=[1, 2, 3])))
        assert client.search("q").results == ()

    def test_pagination_accumulates_then_caps(self) -> None:
        pages = {
            0: [_result(f"http://p1-{i}") for i in range(10)],
            10: [_result(f"http://p2-{i}") for i in range(10)],
        }

        def handler(req: httpx.Request) -> httpx.Response:
            offset = int(req.url.params.get("offset", "0"))
            return httpx.Response(200, json={"results": pages[offset]})

        client = HostedSearchClient(http=_http(handler), config=HostedSearchConfig(page_size=10))
        resp = client.search("q", max_results=15)
        assert len(resp.results) == 15
        assert len(resp.raw["pages"]) == 2

    def test_short_page_stops_pagination(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            offset = int(req.url.params.get("offset", "0"))
            return httpx.Response(
                200, json={"results": [_result("http://a")] if offset == 0 else []}
            )

        client = HostedSearchClient(http=_http(handler), config=HostedSearchConfig(page_size=10))
        resp = client.search("q", max_results=50)
        assert len(resp.results) == 1

    def test_as_of_param_rides_the_request_when_configured(self) -> None:
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["params"] = dict(req.url.params)
            return httpx.Response(200, json={"results": []})

        client = HostedSearchClient(
            http=_http(handler), config=HostedSearchConfig(as_of_param="freshness_max")
        )
        client.search("q", as_of=datetime(2024, 6, 1, 12, tzinfo=UTC))
        assert seen["params"]["freshness_max"] == "2024-06-01"

    def test_as_of_omitted_without_configured_param(self) -> None:
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["params"] = dict(req.url.params)
            return httpx.Response(200, json={"results": []})

        client = HostedSearchClient(http=_http(handler))
        client.search("q", as_of=datetime(2024, 6, 1, tzinfo=UTC))
        assert "freshness_max" not in seen["params"]
        assert "2024-06-01" not in seen["params"].values()

    def test_configured_param_ignored_without_as_of(self) -> None:
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["params"] = dict(req.url.params)
            return httpx.Response(200, json={"results": []})

        client = HostedSearchClient(
            http=_http(handler), config=HostedSearchConfig(as_of_param="freshness_max")
        )
        client.search("q")
        assert "freshness_max" not in seen["params"]

    def test_rejects_bad_max_results(self) -> None:
        client = HostedSearchClient(
            http=_http(lambda _r: httpx.Response(200, json={"results": []}))
        )
        with pytest.raises(ValueError, match="max_results must be"):
            client.search("q", max_results=0)

    def test_server_error_propagates(self) -> None:
        client = HostedSearchClient(
            http=_http(lambda _r: httpx.Response(500, text="boom")),
            config=HostedSearchConfig(),
        )
        with pytest.raises(HttpError):
            client.search("q")

    def test_api_key_header_sent_from_secrets(self) -> None:
        seen: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["auth"] = req.headers.get("X-Api-Key", "")
            return httpx.Response(200, json={"results": []})

        secrets = EnvSecretProvider({"DELPHI_SECRET_SEARCH_KEY": "sk-123"})
        config = HostedSearchConfig(api_key_secret="search-key", api_key_header="X-Api-Key")
        client = HostedSearchClient(http=_http(handler), config=config, secrets=secrets)
        client.search("q")
        assert seen["auth"] == "sk-123"

    def test_no_auth_header_without_secret(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert "Authorization" not in req.headers
            return httpx.Response(200, json={"results": []})

        HostedSearchClient(http=_http(handler)).search("q")
