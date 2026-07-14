"""Hermetic tests for the ForecastBench dataset fetcher (network mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from benchmarks.base import assert_no_leakage
from benchmarks.fetchers.forecastbench_repo import (
    ForecastBenchFetcher,
    map_question,
    map_resolutions,
)
from benchmarks.forecastbench import ForecastBenchAdapter
from common.http.client import HttpClient


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _question(**overrides: Any) -> dict[str, Any]:
    q: dict[str, Any] = {
        "id": "abc",
        "source": "metaculus",
        "question": "Will Y happen?",
        "category": "economics",
        "resolution_criteria": "Resolves per official data.",
        "freeze_datetime": "2026-01-01T00:00:00Z",
        "freeze_datetime_value": 0.4,
        "resolution_date": "2026-06-01T00:00:00Z",
    }
    q.update(overrides)
    return q


class TestMapResolutions:
    def test_indexes_resolved_entries(self) -> None:
        doc = {
            "resolutions": [
                {
                    "id": "abc",
                    "source": "metaculus",
                    "resolved_to": 1,
                    "resolution_date": "2026-06-01T00:00:00Z",
                },
                {"id": "def", "source": "manifold", "resolved": False},
            ]
        }
        resolved = map_resolutions(doc)
        assert set(resolved) == {"metaculus-abc"}
        assert resolved["metaculus-abc"]["resolved_value"] == 1.0

    def test_skips_incomplete_entries(self) -> None:
        doc = {"resolutions": [{"id": "x", "source": "s"}]}  # no value/date
        assert map_resolutions(doc) == {}

    def test_non_sequence_returns_empty(self) -> None:
        assert map_resolutions({"resolutions": None}) == {}


class TestMapQuestion:
    def test_maps_with_resolution(self) -> None:
        resolutions = map_resolutions(
            {
                "resolutions": [
                    {
                        "id": "abc",
                        "source": "metaculus",
                        "resolved_to": 1,
                        "resolution_date": "2026-06-01T00:00:00Z",
                    }
                ]
            }
        )
        record = map_question(_question(), resolutions=resolutions)
        assert record is not None
        assert record["id"] == "metaculus-abc"
        assert record["question"] == "Will Y happen?"
        assert record["as_of"] == "2026-01-01T00:00:00Z"
        assert record["domain"] == "economics"
        assert record["freeze_value"] == 0.4
        assert record["resolved_value"] == 1.0
        assert record["resolved_at"] == "2026-06-01T00:00:00Z"

    def test_open_question_has_no_resolution(self) -> None:
        record = map_question(_question(), resolutions={})
        assert record is not None
        assert "resolved_value" not in record

    def test_default_as_of_used_when_no_freeze(self) -> None:
        q = _question()
        q.pop("freeze_datetime")
        record = map_question(q, default_as_of="2025-12-01T00:00:00Z")
        assert record is not None
        assert record["as_of"] == "2025-12-01T00:00:00Z"

    def test_missing_as_of_skipped(self) -> None:
        q = _question()
        q.pop("freeze_datetime")
        assert map_question(q) is None

    def test_freeze_at_override(self) -> None:
        record = map_question(_question(), freeze_at=datetime(2026, 3, 1, tzinfo=UTC))
        assert record is not None
        assert record["as_of"] == "2026-03-01T00:00:00+00:00"

    def test_missing_text_skipped(self) -> None:
        q = _question()
        q.pop("question")
        assert map_question(q) is None

    def test_na_close_time_dropped(self) -> None:
        q = _question()
        q["resolution_date"] = "N/A"  # ForecastBench sentinel for open markets
        record = map_question(q)
        assert record is not None
        assert "close_time" not in record

    def test_na_freeze_datetime_falls_back_to_default(self) -> None:
        q = _question()
        q["freeze_datetime"] = "N/A"
        record = map_question(q, default_as_of="2025-12-01T00:00:00Z")
        assert record is not None
        assert record["as_of"] == "2025-12-01T00:00:00Z"


class TestForecastBenchFetcher:
    def test_fetch_joins_questions_and_resolutions(self) -> None:
        qdoc = {"forecast_due_date": "2026-01-01T00:00:00Z", "questions": [_question()]}
        rdoc = {
            "resolutions": [
                {
                    "id": "abc",
                    "source": "metaculus",
                    "resolved_to": 1,
                    "resolution_date": "2026-06-01T00:00:00Z",
                }
            ]
        }
        docs = {
            "https://host/questions.json": qdoc,
            "https://host/resolutions.json": rdoc,
        }

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=docs[str(req.url)])

        fetcher = ForecastBenchFetcher(http=_http(handler), base_url="https://host")
        records = fetcher.fetch(question_set="questions.json", resolution_set="resolutions.json")
        adapter = ForecastBenchAdapter.from_records(records)
        assert_no_leakage(adapter.questions(), adapter.resolutions())
        assert adapter.questions()[0].question_id == "forecastbench:metaculus-abc"
        assert adapter.resolutions()[0].resolved_value == 1.0

    def test_fetch_without_resolution_set(self) -> None:
        qdoc = {"questions": [_question()]}

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=qdoc)

        fetcher = ForecastBenchFetcher(http=_http(handler))
        records = fetcher.fetch(question_set="https://host/q.json")
        assert len(records) == 1
        assert "resolved_value" not in records[0]

    def test_absolute_url_passthrough(self) -> None:
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["url"] = str(req.url)
            return httpx.Response(200, json={"questions": []})

        fetcher = ForecastBenchFetcher(http=_http(handler), base_url="https://ignored")
        fetcher.fetch(question_set="https://elsewhere/q.json")
        assert seen["url"] == "https://elsewhere/q.json"
