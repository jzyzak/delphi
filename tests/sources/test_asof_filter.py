"""Leakage + parsing tests for the as-of filter (C3.2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sources.asof_filter import filter_as_of, parse_knowledge_time
from sources.providers.hosted import HostedSearchResult

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _result(url: str, date: str | None) -> HostedSearchResult:
    return HostedSearchResult(url=url, content=f"c-{url}", published_date=date, score=0.3)


class TestParseKnowledgeTime:
    def test_iso_with_offset(self) -> None:
        assert parse_knowledge_time("2024-01-01T00:00:00+00:00") == datetime(2024, 1, 1, tzinfo=UTC)

    def test_trailing_z(self) -> None:
        assert parse_knowledge_time("2024-01-01T12:00:00Z") == datetime(2024, 1, 1, 12, tzinfo=UTC)

    def test_date_only_assumed_utc(self) -> None:
        assert parse_knowledge_time("2024-01-01") == datetime(2024, 1, 1, tzinfo=UTC)

    def test_naive_assumed_utc(self) -> None:
        assert parse_knowledge_time("2024-01-01T06:00:00") == datetime(2024, 1, 1, 6, tzinfo=UTC)

    @pytest.mark.parametrize("bad", [None, "", "  ", "not-a-date", "yesterday"])
    def test_unparseable_returns_none(self, bad: str | None) -> None:
        assert parse_knowledge_time(bad) is None


class TestFilterAsOf:
    def test_drops_post_as_of_leakage(self) -> None:
        results = [
            _result("http://before", "2024-01-01"),
            _result("http://after", "2024-12-31"),  # leaked
        ]
        evidence = filter_as_of(results, as_of=AS_OF, provider="hosted", query="q")
        assert [e.source_id for e in evidence] == ["http://before"]
        assert all(e.knowledge_time <= AS_OF for e in evidence)

    def test_drops_undated_as_unsafe(self) -> None:
        results = [_result("http://dated", "2024-01-01"), _result("http://undated", None)]
        evidence = filter_as_of(results, as_of=AS_OF, provider="hosted")
        assert [e.source_id for e in evidence] == ["http://dated"]

    def test_maps_fields_and_at_ceiling_is_kept(self) -> None:
        results = [_result("http://x", "2024-06-01T00:00:00+00:00")]
        (ev,) = filter_as_of(results, as_of=AS_OF, provider="tavily", query="qq")
        assert ev.snippet == "c-http://x"
        assert ev.source == "tavily"
        assert ev.query == "qq"
        assert ev.knowledge_time == AS_OF  # boundary is inclusive

    def test_deterministic_order(self) -> None:
        results = [
            _result("http://b", "2024-02-01"),
            _result("http://a", "2024-01-01"),
            _result("http://c", "2024-01-01"),
        ]
        evidence = filter_as_of(results, as_of=AS_OF, provider="hosted")
        assert [e.source_id for e in evidence] == ["http://a", "http://c", "http://b"]

    def test_empty(self) -> None:
        assert filter_as_of([], as_of=AS_OF, provider="hosted") == ()

    def test_title_used_when_content_empty(self) -> None:
        r = HostedSearchResult(
            url="http://x", title="headline", content="", published_date="2024-01-01"
        )
        (ev,) = filter_as_of([r], as_of=AS_OF, provider="hosted")
        assert ev.snippet == "headline"
