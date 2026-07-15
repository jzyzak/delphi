"""Tests for the Wikipedia revision-history provider (hermetic; network mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from common.http.client import HttpClient
from core.forecast.search import AsOfSearcher
from sources.providers.wikipedia import WikipediaAsOfSearcher, WikipediaConfig
from sources.snapshot import InMemorySnapshotStore

AS_OF = datetime(2024, 6, 1, 12, 30, 45, tzinfo=UTC)

_SEARCH_HITS = [
    {"pageid": 123, "title": "Foo"},
    {"pageid": 456, "title": "Bar"},
]


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _revision_page(
    *, pageid: int, title: str, revid: int, timestamp: str, content: str
) -> dict[str, Any]:
    return {
        "pageid": pageid,
        "title": title,
        "revisions": [
            {
                "revid": revid,
                "parentid": revid - 1,
                "timestamp": timestamp,
                "slots": {"main": {"content": content}},
            }
        ],
    }


def _default_handler(seen: dict[str, Any]) -> Any:
    """Search returns Foo + Bar; Foo has a pinned revision, Bar predates nothing."""

    def handler(req: httpx.Request) -> httpx.Response:
        params = dict(req.url.params)
        seen.setdefault("requests", []).append(params)
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": _SEARCH_HITS}})
        if params.get("prop") == "revisions":
            if params["titles"] == "Foo":
                page = _revision_page(
                    pageid=123,
                    title="Foo",
                    revid=999,
                    timestamp="2024-05-01T10:00:00Z",
                    content="Foo article text. " * 50,
                )
                return httpx.Response(200, json={"query": {"pages": [page]}})
            # Bar has no revision at/before as_of (page did not exist yet).
            return httpx.Response(200, json={"query": {"pages": [{"pageid": 456, "title": "Bar"}]}})
        raise AssertionError(f"unexpected request params: {params}")

    return handler


def _searcher(
    handler: Any,
    *,
    store: InMemorySnapshotStore | None = None,
    max_results: int = 5,
    config: WikipediaConfig | None = None,
) -> WikipediaAsOfSearcher:
    return WikipediaAsOfSearcher(
        http=_http(handler),
        config=config,
        snapshot_store=store if store is not None else InMemorySnapshotStore(),
        max_results=max_results,
    )


class TestWikipediaAsOfSearcher:
    def test_is_asof_searcher(self) -> None:
        seen: dict[str, Any] = {}
        assert isinstance(_searcher(_default_handler(seen)), AsOfSearcher)

    def test_search_request_params(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen), max_results=3)
        searcher.as_of_search("foo question", as_of=AS_OF)
        search_req = seen["requests"][0]
        assert search_req["action"] == "query"
        assert search_req["list"] == "search"
        assert search_req["srsearch"] == "foo question"
        assert search_req["srlimit"] == "3"
        assert search_req["format"] == "json"
        assert search_req["formatversion"] == "2"

    def test_revision_lookup_is_pinned_at_as_of(self) -> None:
        """The revision query itself is bounded: rvstart=as_of, rvdir=older."""
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen))
        searcher.as_of_search("q", as_of=AS_OF)
        rev_reqs = [r for r in seen["requests"] if r.get("prop") == "revisions"]
        assert [r["titles"] for r in rev_reqs] == ["Foo", "Bar"]
        for req in rev_reqs:
            assert req["rvstart"] == "2024-06-01T12:30:45Z"
            assert req["rvdir"] == "older"
            assert req["rvlimit"] == "1"
            assert req["rvprop"] == "ids|timestamp|content"
            assert req["rvslots"] == "main"

    def test_knowledge_time_is_revision_timestamp_and_leq_as_of(self) -> None:
        """AS-OF/LEAKAGE: knowledge_time == the pinned revision timestamp <= as_of."""
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen))
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert len(evidence) == 1
        assert evidence[0].knowledge_time == datetime(2024, 5, 1, 10, 0, tzinfo=UTC)
        assert all(e.knowledge_time <= AS_OF for e in evidence)

    def test_evidence_shape(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen))
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        item = evidence[0]
        assert item.source == "wikipedia"
        assert item.source_id == "wikipedia:123:999"
        assert item.snippet.startswith("Foo article text.")
        assert len(item.snippet) <= 600
        assert item.query == "q"
        assert item.score == 1.0  # top-ranked search hit

    def test_page_without_revision_at_or_before_as_of_is_skipped(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen))
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert {e.source_id for e in evidence} == {"wikipedia:123:999"}  # Bar skipped

    def test_leakage_revision_after_as_of_is_dropped_by_filter(self) -> None:
        """AS-OF/LEAKAGE belt-and-suspenders: a post-as_of timestamp never survives."""

        def handler(req: httpx.Request) -> httpx.Response:
            params = dict(req.url.params)
            if params.get("list") == "search":
                return httpx.Response(
                    200, json={"query": {"search": [{"pageid": 1, "title": "Sloppy"}]}}
                )
            page = _revision_page(
                pageid=1,
                title="Sloppy",
                revid=2,
                timestamp="2024-12-31T00:00:00Z",  # after as_of despite rvstart
                content="leaked",
            )
            return httpx.Response(200, json={"query": {"pages": [page]}})

        searcher = _searcher(handler)
        assert searcher.as_of_search("q", as_of=AS_OF) == ()

    def test_snapshot_first_replays_without_second_network_call(self) -> None:
        seen: dict[str, Any] = {}
        store = InMemorySnapshotStore()
        searcher = _searcher(_default_handler(seen), store=store)

        first = searcher.as_of_search("q", as_of=AS_OF)
        second = searcher.as_of_search("q", as_of=AS_OF)

        assert first == second
        assert len(seen["requests"]) == 3  # 1 search + 2 revision lookups, no replay traffic
        assert len(store) == 1
        snapshot = next(iter(store))
        assert snapshot.provider == "wikipedia"
        assert "search" in snapshot.raw["pages"][0]
        assert len(snapshot.raw["pages"][0]["revisions"]) == 2

    def test_distinct_as_of_triggers_new_fetch(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen))
        searcher.as_of_search("q", as_of=AS_OF)
        searcher.as_of_search("q", as_of=datetime(2024, 7, 1, tzinfo=UTC))
        assert len(seen["requests"]) == 6

    def test_naive_as_of_rejected(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_default_handler(seen))
        with pytest.raises(ValueError, match="Naive"):
            searcher.as_of_search("q", as_of=datetime(2024, 6, 1))  # noqa: DTZ001

    def test_snippet_truncated_to_config_chars(self) -> None:
        seen: dict[str, Any] = {}
        config = WikipediaConfig(snippet_chars=10)
        searcher = _searcher(_default_handler(seen), config=config)
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert evidence[0].snippet == "Foo articl"

    def test_empty_revision_content_falls_back_to_title(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            params = dict(req.url.params)
            if params.get("list") == "search":
                return httpx.Response(
                    200, json={"query": {"search": [{"pageid": 7, "title": "Stub"}]}}
                )
            page = _revision_page(
                pageid=7, title="Stub", revid=8, timestamp="2024-01-01T00:00:00Z", content=""
            )
            return httpx.Response(200, json={"query": {"pages": [page]}})

        evidence = _searcher(handler).as_of_search("q", as_of=AS_OF)
        assert evidence[0].snippet == "Stub"

    def test_no_search_hits_returns_empty(self) -> None:
        handler = lambda _r: httpx.Response(200, json={"query": {"search": []}})  # noqa: E731
        assert _searcher(handler).as_of_search("q", as_of=AS_OF) == ()

    @pytest.mark.parametrize(
        "payload",
        [
            ["unexpected"],
            {},
            {"query": "nope"},
            {"query": {"search": "nope"}},
            {"query": {"search": ["junk", {"pageid": 1}, {"title": ""}]}},
        ],
    )
    def test_tolerates_malformed_search_payloads(self, payload: Any) -> None:
        searcher = _searcher(lambda _r: httpx.Response(200, json=payload))
        assert searcher.as_of_search("q", as_of=AS_OF) == ()

    @pytest.mark.parametrize(
        "rev_payload",
        [
            ["unexpected"],
            {},
            {"query": "nope"},
            {"query": {"pages": "nope"}},
            {"query": {"pages": []}},
            {"query": {"pages": ["junk"]}},
            {"query": {"pages": [{"pageid": 1, "revisions": "nope"}]}},
            {"query": {"pages": [{"pageid": 1, "revisions": ["junk"]}]}},
            {"query": {"pages": [{"pageid": 1, "revisions": [{"revid": 2}]}]}},  # no timestamp
            {  # missing pageid
                "query": {
                    "pages": [{"revisions": [{"revid": 2, "timestamp": "2024-01-01T00:00:00Z"}]}]
                }
            },
            {  # missing revid
                "query": {
                    "pages": [{"pageid": 1, "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}]}]
                }
            },
        ],
    )
    def test_tolerates_malformed_revision_payloads(self, rev_payload: Any) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            params = dict(req.url.params)
            if params.get("list") == "search":
                return httpx.Response(
                    200, json={"query": {"search": [{"pageid": 1, "title": "Foo"}]}}
                )
            return httpx.Response(200, json=rev_payload)

        assert _searcher(handler).as_of_search("q", as_of=AS_OF) == ()

    def test_missing_slots_yields_title_snippet(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            params = dict(req.url.params)
            if params.get("list") == "search":
                return httpx.Response(
                    200, json={"query": {"search": [{"pageid": 1, "title": "Slotless"}]}}
                )
            page = {
                "pageid": 1,
                "title": "Slotless",
                "revisions": [{"revid": 2, "timestamp": "2024-01-01T00:00:00Z"}],
            }
            return httpx.Response(200, json={"query": {"pages": [page]}})

        evidence = _searcher(handler).as_of_search("q", as_of=AS_OF)
        assert evidence[0].snippet == "Slotless"

    def test_rejects_bad_max_results_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_results"):
            _searcher(lambda _r: httpx.Response(200, json={}), max_results=0)

    def test_default_snapshot_store_is_in_memory(self) -> None:
        searcher = WikipediaAsOfSearcher(
            http=_http(lambda _r: httpx.Response(200, json={"query": {"search": []}}))
        )
        assert searcher.as_of_search("q", as_of=AS_OF) == ()

    def test_rank_scores_descend_across_hits(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            params = dict(req.url.params)
            if params.get("list") == "search":
                hits = [{"pageid": i, "title": f"T{i}"} for i in (1, 2)]
                return httpx.Response(200, json={"query": {"search": hits}})
            pageid = 1 if params["titles"] == "T1" else 2
            page = _revision_page(
                pageid=pageid,
                title=params["titles"],
                revid=pageid * 10,
                timestamp="2024-01-01T00:00:00Z",
                content="x",
            )
            return httpx.Response(200, json={"query": {"pages": [page]}})

        evidence = _searcher(handler).as_of_search("q", as_of=AS_OF)
        scores = {e.source_id: e.score for e in evidence}
        assert scores["wikipedia:1:10"] == 1.0
        assert scores["wikipedia:2:20"] == 0.5
