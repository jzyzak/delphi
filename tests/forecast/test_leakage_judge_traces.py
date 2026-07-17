"""Unit tests for the leakage-judge trace builders (core/forecast/leakage_judge)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

from core.forecast.leakage_judge import TraceComponent, trace_from_evidence
from core.forecast.search import Evidence

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _evidence(snippet: str, *, source_id: str = "http://a", score: float = 0.5) -> Evidence:
    return Evidence(
        snippet=snippet,
        source="tavily",
        source_id=source_id,
        knowledge_time=datetime(2024, 5, 1, tzinfo=UTC),
        score=score,
        query="q",
    )


class TestTraceFromEvidence:
    def test_builds_search_trace_pinned_at_ceiling(self) -> None:
        trace = trace_from_evidence((_evidence("snippet-a"),), as_of=AS_OF, forecast_id="q-1")
        assert trace.component is TraceComponent.SEARCH
        assert trace.as_of == AS_OF
        assert trace.forecast_id == "q-1"
        assert trace.metadata == {"n_evidence": 1}

    def test_text_carries_raw_snippets_and_provenance(self) -> None:
        trace = trace_from_evidence(
            (
                _evidence("first snippet", source_id="http://a"),
                _evidence("second snippet", source_id="http://b", score=0.9),
            ),
            as_of=AS_OF,
        )
        payload = json.loads(trace.text)
        assert [item["snippet"] for item in payload] == ["first snippet", "second snippet"]
        assert payload[0]["source"] == "tavily"
        assert payload[0]["source_id"] == "http://a"
        assert payload[0]["knowledge_time"] == "2024-05-01T00:00:00+00:00"
        assert payload[1]["score"] == 0.9
        assert payload[0]["query"] == "q"

    def test_empty_evidence_yields_empty_payload(self) -> None:
        trace = trace_from_evidence((), as_of=AS_OF)
        assert json.loads(trace.text) == []
        assert trace.metadata == {"n_evidence": 0}

    def test_as_of_normalized_to_utc(self) -> None:
        trace = trace_from_evidence(
            (_evidence("s"),),
            as_of=datetime(2024, 6, 1, 3, 0, tzinfo=timezone(timedelta(hours=3))),
        )
        assert trace.as_of == datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
