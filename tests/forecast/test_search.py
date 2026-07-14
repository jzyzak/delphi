"""Unit tests for domain-agnostic as-of search seam (§8)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.forecast.search import Evidence, FixtureAsOfSearch

T_AS_OF = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)
T_EARLY = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)


def _evidence(*, snippet: str = "test snippet", query: str = "test query") -> Evidence:
    return Evidence(
        snippet=snippet,
        source="fda",
        source_id="FDA-1",
        knowledge_time=T_EARLY,
        score=0.8,
        query=query,
    )


class TestEvidence:
    def test_happy_path(self) -> None:
        ev = _evidence()
        assert ev.source == "fda"
        assert ev.knowledge_time.tzinfo is not None

    def test_failure_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            Evidence(
                snippet="x",
                source="web",
                source_id="W-1",
                knowledge_time=datetime(2024, 1, 1, 12, 0),
                score=0.5,
            )

    def test_failure_negative_score_raises(self) -> None:
        with pytest.raises(ValueError):
            Evidence(
                snippet="x",
                source="fda",
                source_id="F-1",
                knowledge_time=T_EARLY,
                score=-0.1,
            )


class TestFixtureAsOfSearch:
    def test_happy_path_returns_configured_response(self) -> None:
        ev = _evidence(snippet="approval base rate")
        search = FixtureAsOfSearch(
            responses={"approval base rate": (ev,)},
        )
        results = search.as_of_search("approval base rate", as_of=T_AS_OF)
        assert len(results) == 1
        assert results[0].snippet == "approval base rate"
        assert search.call_count == 1
        assert search.queries == [("approval base rate", T_AS_OF)]

    def test_boundary_case_insensitive_lookup(self) -> None:
        ev = _evidence()
        search = FixtureAsOfSearch(responses={"my query": (ev,)})
        results = search.as_of_search("  MY QUERY  ", as_of=T_AS_OF)
        assert len(results) == 1

    def test_failure_unknown_query_returns_default(self) -> None:
        default = _evidence(snippet="default")
        search = FixtureAsOfSearch(default=(default,))
        results = search.as_of_search("unknown", as_of=T_AS_OF)
        assert results == (default,)

    def test_failure_empty_default_returns_empty(self) -> None:
        search = FixtureAsOfSearch()
        assert search.as_of_search("q", as_of=T_AS_OF) == ()
