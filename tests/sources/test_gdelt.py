"""Tests for the GDELT DOC 2.0 historical provider (hermetic; network mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from common.http.client import HttpClient
from core.forecast.search import AsOfSearcher
from sources.providers.gdelt import GdeltAsOfSearcher, GdeltConfig, parse_seendate
from sources.snapshot import InMemorySnapshotStore

AS_OF = datetime(2024, 6, 1, 12, 30, 45, tzinfo=UTC)


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _articles() -> list[dict[str, Any]]:
    return [
        {
            "url": "http://a.example/1",
            "title": "Old article A",
            "seendate": "20240115T083000Z",
            "domain": "a.example",
            "language": "English",
            "sourcecountry": "US",
        },
        {"url": "http://b.example/2", "title": "Old article B", "seendate": "20240520T000000Z"},
        # Post-as-of despite the enddatetime bound (simulated API sloppiness).
        {
            "url": "http://leaky.example/3",
            "title": "From the future",
            "seendate": "20240701T000000Z",
        },
        {"url": "http://undated.example/4", "title": "No seendate"},
        {"url": "http://baddate.example/5", "title": "Bad seendate", "seendate": "not-a-date"},
    ]


def _searcher(
    handler: Any,
    *,
    store: InMemorySnapshotStore | None = None,
    max_results: int = 10,
) -> GdeltAsOfSearcher:
    return GdeltAsOfSearcher(
        http=_http(handler),
        snapshot_store=store if store is not None else InMemorySnapshotStore(),
        max_results=max_results,
    )


def _ok_handler(articles: list[dict[str, Any]], seen: dict[str, Any] | None = None) -> Any:
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["url"] = str(req.url).split("?")[0]
            seen["params"] = dict(req.url.params)
            seen["n"] = seen.get("n", 0) + 1
        return httpx.Response(200, json={"articles": articles})

    return handler


class TestParseSeendate:
    def test_parses_gdelt_format_to_iso_utc(self) -> None:
        assert parse_seendate("20240115T083000Z") == "2024-01-15T08:30:00+00:00"

    @pytest.mark.parametrize("raw", [None, "", "   ", "not-a-date", "2024-01-15", 20240115])
    def test_missing_or_malformed_is_none(self, raw: Any) -> None:
        assert parse_seendate(raw) is None


class TestGdeltAsOfSearcher:
    def test_is_asof_searcher(self) -> None:
        searcher = _searcher(_ok_handler([]))
        assert isinstance(searcher, AsOfSearcher)

    def test_request_params_are_bounded_at_as_of(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_ok_handler(_articles(), seen), max_results=7)
        searcher.as_of_search("election outcome", as_of=AS_OF)

        assert seen["url"] == "https://api.gdeltproject.org/api/v2/doc/doc"
        params = seen["params"]
        assert params["query"] == "election outcome"
        assert params["mode"] == "artlist"
        assert params["format"] == "json"
        assert params["maxrecords"] == "7"
        assert params["sort"] == "hybridrel"
        # enddatetime is the as-of ceiling as YYYYMMDDHHMMSS UTC.
        assert params["enddatetime"] == "20240601123045"
        # startdatetime reaches back lookback_days (default 90) from the ceiling.
        assert params["startdatetime"] == "20240303123045"

    def test_as_of_converted_to_utc_before_formatting(self) -> None:
        from datetime import timedelta, timezone

        seen: dict[str, Any] = {}
        searcher = _searcher(_ok_handler([], seen))
        offset = timezone(timedelta(hours=2))
        searcher.as_of_search("q", as_of=datetime(2024, 6, 1, 14, 30, 45, tzinfo=offset))
        assert seen["params"]["enddatetime"] == "20240601123045"

    def test_naive_as_of_rejected(self) -> None:
        searcher = _searcher(_ok_handler([]))
        with pytest.raises(ValueError, match="Naive"):
            searcher.as_of_search("q", as_of=datetime(2024, 6, 1))  # noqa: DTZ001

    def test_maps_articles_to_evidence(self) -> None:
        searcher = _searcher(_ok_handler(_articles()))
        evidence = searcher.as_of_search("q", as_of=AS_OF)

        by_id = {e.source_id: e for e in evidence}
        assert set(by_id) == {"http://a.example/1", "http://b.example/2"}
        first = by_id["http://a.example/1"]
        # GDELT gives no snippet: the title is the snippet.
        assert first.snippet == "Old article A"
        assert first.source == "gdelt"
        assert first.query == "q"
        assert first.knowledge_time == datetime(2024, 1, 15, 8, 30, tzinfo=UTC)
        assert all(e.score > 0.0 for e in evidence)

    def test_rank_based_scores_descend_with_position(self) -> None:
        articles = [
            {"url": f"http://x/{i}", "title": f"t{i}", "seendate": "20240101T000000Z"}
            for i in range(4)
        ]
        searcher = _searcher(_ok_handler(articles))
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        score_by_id = {e.source_id: e.score for e in evidence}
        assert score_by_id["http://x/0"] == 1.0
        assert (
            score_by_id["http://x/0"]
            > score_by_id["http://x/1"]
            > score_by_id["http://x/2"]
            > score_by_id["http://x/3"]
        )

    def test_leakage_post_as_of_article_is_dropped(self) -> None:
        """AS-OF/LEAKAGE: a seendate after as_of never survives, even if GDELT leaks it."""
        searcher = _searcher(_ok_handler(_articles()))
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert "http://leaky.example/3" not in {e.source_id for e in evidence}
        assert all(e.knowledge_time <= AS_OF for e in evidence)

    def test_leakage_undated_and_baddated_articles_are_dropped(self) -> None:
        searcher = _searcher(_ok_handler(_articles()))
        ids = {e.source_id for e in searcher.as_of_search("q", as_of=AS_OF)}
        assert "http://undated.example/4" not in ids
        assert "http://baddate.example/5" not in ids

    def test_snapshot_first_replays_without_second_network_call(self) -> None:
        seen: dict[str, Any] = {}
        store = InMemorySnapshotStore()
        searcher = _searcher(_ok_handler(_articles(), seen), store=store)

        first = searcher.as_of_search("q", as_of=AS_OF)
        second = searcher.as_of_search("q", as_of=AS_OF)

        assert first == second
        assert seen["n"] == 1  # second call served from the snapshot
        assert len(store) == 1
        snapshot = next(iter(store))
        assert snapshot.provider == "gdelt"
        assert snapshot.raw == {"pages": [{"articles": _articles()}]}

    def test_distinct_as_of_triggers_new_fetch(self) -> None:
        seen: dict[str, Any] = {}
        searcher = _searcher(_ok_handler(_articles(), seen))
        searcher.as_of_search("q", as_of=AS_OF)
        searcher.as_of_search("q", as_of=datetime(2024, 7, 1, tzinfo=UTC))
        assert seen["n"] == 2

    def test_max_results_caps_mapped_articles(self) -> None:
        articles = [
            {"url": f"http://x/{i}", "title": f"t{i}", "seendate": "20240101T000000Z"}
            for i in range(5)
        ]
        searcher = _searcher(_ok_handler(articles), max_results=2)
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert {e.source_id for e in evidence} == {"http://x/0", "http://x/1"}

    def test_tolerates_non_dict_payload(self) -> None:
        searcher = _searcher(lambda _r: httpx.Response(200, json=["unexpected"]))
        assert searcher.as_of_search("q", as_of=AS_OF) == ()

    def test_tolerates_non_list_articles(self) -> None:
        searcher = _searcher(lambda _r: httpx.Response(200, json={"articles": "nope"}))
        assert searcher.as_of_search("q", as_of=AS_OF) == ()

    def test_tolerates_non_dict_article_entries(self) -> None:
        articles: list[Any] = [
            "junk",
            {"url": "http://ok", "title": "ok", "seendate": "20240101T000000Z"},
        ]
        searcher = _searcher(lambda _r: httpx.Response(200, json={"articles": articles}))
        evidence = searcher.as_of_search("q", as_of=AS_OF)
        assert [e.source_id for e in evidence] == ["http://ok"]

    def test_rejects_bad_max_results_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_results"):
            _searcher(_ok_handler([]), max_results=0)

    def test_search_rejects_bad_max_results(self) -> None:
        searcher = _searcher(_ok_handler([]))
        with pytest.raises(ValueError, match="max_results"):
            searcher.search("q", max_results=0, as_of=AS_OF)

    def test_custom_config_is_used(self) -> None:
        seen: dict[str, Any] = {}
        config = GdeltConfig(
            base_url="https://gdelt.mirror.example/doc", sort="datedesc", lookback_days=30
        )
        searcher = GdeltAsOfSearcher(
            http=_http(_ok_handler([], seen)),
            config=config,
            snapshot_store=InMemorySnapshotStore(),
        )
        assert searcher.config == config
        searcher.as_of_search("q", as_of=AS_OF)
        assert seen["url"] == "https://gdelt.mirror.example/doc"
        assert seen["params"]["sort"] == "datedesc"
        assert seen["params"]["startdatetime"] == "20240502123045"

    def test_default_snapshot_store_is_in_memory(self) -> None:
        searcher = GdeltAsOfSearcher(http=_http(_ok_handler([])))
        assert searcher.as_of_search("q", as_of=AS_OF) == ()
