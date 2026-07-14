"""Hermetic tests for the Metaculus API fetcher (network mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from benchmarks.fetchers.metaculus_api import (
    METACULUS_API_TOKEN_SECRET,
    MetaculusFetcher,
    map_post,
)
from benchmarks.metaculus import MetaculusAdapter
from common.http.client import HttpClient
from common.secrets import EnvSecretProvider


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _binary_post(**overrides: Any) -> dict[str, Any]:
    post: dict[str, Any] = {
        "id": 101,
        "title": "Will X ship in 2026?",
        "projects": {"category": [{"slug": "technology", "name": "Technology"}]},
        "question": {
            "id": 555,
            "type": "binary",
            "description": "Resolves YES if X ships.",
            "open_time": "2026-01-01T00:00:00Z",
            "scheduled_close_time": "2026-06-01T00:00:00Z",
            "actual_resolve_time": "2026-07-01T00:00:00Z",
            "resolution": "yes",
            "aggregations": {"recency_weighted": {"latest": {"centers": [0.63]}}},
        },
    }
    post.update(overrides)
    return post


class TestMapPost:
    def test_maps_binary_resolved_question(self) -> None:
        record = map_post(_binary_post())
        assert record is not None
        assert record["id"] == 101
        assert record["title"] == "Will X ship in 2026?"
        assert record["as_of"] == "2026-01-01T00:00:00Z"
        assert record["domain"] == "technology"
        assert record["community"] == 0.63
        assert record["resolution"] == 1.0
        assert record["resolved_at"] == "2026-07-01T00:00:00Z"

    def test_resolution_no_maps_to_zero(self) -> None:
        post = _binary_post()
        post["question"]["resolution"] = "no"
        record = map_post(post)
        assert record is not None
        assert record["resolution"] == 0.0

    def test_annulled_resolution_omitted(self) -> None:
        post = _binary_post()
        post["question"]["resolution"] = "annulled"
        record = map_post(post)
        assert record is not None
        assert "resolution" not in record
        assert "resolved_at" not in record

    def test_non_binary_skipped_by_default(self) -> None:
        post = _binary_post()
        post["question"]["type"] = "numeric"
        assert map_post(post) is None

    def test_non_binary_kept_when_allowed(self) -> None:
        post = _binary_post()
        post["question"]["type"] = "numeric"
        record = map_post(post, binary_only=False)
        assert record is not None
        assert record["question_type"] == "numeric"

    def test_missing_as_of_skipped(self) -> None:
        post = _binary_post()
        post["question"].pop("open_time")
        assert map_post(post) is None

    def test_freeze_at_override(self) -> None:
        record = map_post(_binary_post(), freeze_at=datetime(2026, 2, 1, tzinfo=UTC))
        assert record is not None
        assert record["as_of"] == "2026-02-01T00:00:00+00:00"

    def test_missing_community_omitted(self) -> None:
        post = _binary_post()
        post["question"].pop("aggregations")
        record = map_post(post)
        assert record is not None
        assert "community" not in record


class TestMetaculusFetcher:
    def test_fetch_maps_and_feeds_adapter(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [_binary_post()], "next": None})

        fetcher = MetaculusFetcher(http=_http(handler))
        records = fetcher.fetch()
        assert len(records) == 1
        adapter = MetaculusAdapter.from_records(records)
        assert adapter.questions()[0].question_id == "metaculus:101"
        assert adapter.resolutions()[0].resolved_value == 1.0

    def test_fetch_follows_pagination(self) -> None:
        pages = {
            "https://www.metaculus.com/api/posts/": {
                "results": [_binary_post(id=1)],
                "next": "https://www.metaculus.com/api/posts/?cursor=2",
            },
            "https://www.metaculus.com/api/posts/?cursor=2": {
                "results": [_binary_post(id=2)],
                "next": None,
            },
        }

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=pages[str(req.url)])

        fetcher = MetaculusFetcher(http=_http(handler))
        records = fetcher.fetch(max_pages=5)
        assert [r["id"] for r in records] == [1, 2]

    def test_fetch_respects_max_pages(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [_binary_post()],
                    "next": "https://www.metaculus.com/api/posts/?c",
                },
            )

        fetcher = MetaculusFetcher(http=_http(handler))
        records = fetcher.fetch(max_pages=1)
        assert len(records) == 1

    def test_auth_header_sent_when_token_present(self) -> None:
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["auth"] = req.headers.get("authorization")
            return httpx.Response(200, json={"results": [], "next": None})

        secrets = EnvSecretProvider(
            {"DELPHI_SECRET_METACULUS_API_TOKEN": "tok-abc"},
        )
        fetcher = MetaculusFetcher(http=_http(handler), secrets=secrets)
        fetcher.fetch()
        assert seen["auth"] == "Token tok-abc"

    def test_missing_token_tolerated(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.headers.get("authorization") is None
            return httpx.Response(200, json={"results": [], "next": None})

        fetcher = MetaculusFetcher(
            http=_http(handler),
            secrets=EnvSecretProvider({}),
            api_token_secret=METACULUS_API_TOKEN_SECRET,
        )
        assert fetcher.fetch() == []

    def test_invalid_max_pages_raises(self) -> None:
        fetcher = MetaculusFetcher(http=_http(lambda _r: httpx.Response(200, json={})))
        try:
            fetcher.fetch(max_pages=0)
        except ValueError as exc:
            assert "max_pages" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError")
