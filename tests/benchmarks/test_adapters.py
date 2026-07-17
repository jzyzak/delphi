"""Tests for the concrete benchmark adapters (C7.2-C7.5)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmarks.base import assert_no_leakage, scored_records
from benchmarks.forecastbench import ForecastBenchAdapter
from benchmarks.live import LiveHarvestAdapter
from benchmarks.market_consensus import MarketConsensusAdapter, consensus_baseline
from benchmarks.metaculus import MetaculusAdapter
from evaluation.baselines import MARKET_CONSENSUS

_AS_OF = "2026-01-01T00:00:00Z"
_RESOLVED = "2026-06-01T00:00:00Z"


class TestForecastBench:
    def test_load_map_and_no_leakage(self) -> None:
        adapter = ForecastBenchAdapter.from_records(
            [
                {
                    "id": "q1",
                    "question": "Will X happen?",
                    "as_of": _AS_OF,
                    "domain": "tech",
                    "close_time": "2026-05-01T00:00:00Z",
                    "resolved_value": 1.0,
                    "resolved_at": _RESOLVED,
                },
                {"id": "q2", "question": "Open one?", "as_of": _AS_OF},
            ]
        )
        assert adapter.name == "forecastbench"
        assert len(adapter.questions()) == 2
        assert len(adapter.resolutions()) == 1
        assert_no_leakage(adapter.questions(), adapter.resolutions())
        q = adapter.questions()[0]
        assert q.question_id == "forecastbench:q1"
        assert q.domain == "tech"
        assert q.close_time is not None

    def test_freeze_value_lands_in_metadata(self) -> None:
        adapter = ForecastBenchAdapter.from_records(
            [
                {"id": "q1", "question": "Priced?", "as_of": _AS_OF, "freeze_value": 0.62},
                {"id": "q2", "question": "Unpriced?", "as_of": _AS_OF},
            ]
        )
        priced, unpriced = adapter.questions()
        assert priced.metadata["freeze_value"] == 0.62
        assert "freeze_value" not in unpriced.metadata


class TestMetaculus:
    def test_map_and_community_prediction(self) -> None:
        adapter = MetaculusAdapter.from_records(
            [
                {
                    "id": 42,
                    "title": "Will Y resolve YES?",
                    "as_of": _AS_OF,
                    "community": 0.7,
                    "resolution": 1.0,
                    "resolved_at": _RESOLVED,
                }
            ]
        )
        assert adapter.name == "metaculus"
        q = adapter.questions()[0]
        assert q.question_id == "metaculus:42"
        assert q.metadata["community_prediction"] == pytest.approx(0.7)
        assert adapter.resolutions()[0].resolved_value == 1.0

    def test_no_community_no_resolution(self) -> None:
        adapter = MetaculusAdapter.from_records([{"id": 1, "title": "Open?", "as_of": _AS_OF}])
        assert "community_prediction" not in adapter.questions()[0].metadata
        assert adapter.resolutions() == ()


class TestMarketConsensus:
    def test_consensus_baseline_scores_through_harness(self) -> None:
        adapter = MarketConsensusAdapter.from_records(
            [
                {
                    "id": "m1",
                    "question": "Market resolves YES?",
                    "as_of": _AS_OF,
                    "price": 0.65,
                    "resolved_value": 1.0,
                    "resolved_at": _RESOLVED,
                }
            ]
        )
        assert adapter.name == "market_consensus"
        baseline = consensus_baseline(adapter)
        assert baseline.name == MARKET_CONSENSUS
        assert baseline.predict("market_consensus:m1") == pytest.approx(0.65)
        # The same ids flow into scored records for the model side.
        records = scored_records({"market_consensus:m1": 0.9}, adapter)
        assert len(records) == 1

    def test_baseline_skips_questions_without_price(self) -> None:
        adapter = MarketConsensusAdapter.from_records(
            [{"id": "m1", "question": "q", "as_of": _AS_OF, "price": 0.5}]
        )
        # Ask for a price key that no question carries -> empty baseline.
        baseline = consensus_baseline(adapter, price_key="nonexistent")
        assert baseline.predictions == {}


class TestLiveHarvest:
    def test_pins_to_harvest_time_and_dedupes(self) -> None:
        harvest_time = datetime(2026, 7, 1, tzinfo=UTC)
        records = [
            {"id": "a", "question": "Will A?"},
            {"id": "b", "question": "Will B?"},
        ]
        adapter = LiveHarvestAdapter.harvest(
            records, harvest_time=harvest_time, seen_ids={"live:a"}
        )
        assert adapter.name == "live"
        ids = [q.question_id for q in adapter.questions()]
        assert ids == ["live:b"]  # 'a' deduped
        assert adapter.questions()[0].as_of == harvest_time
        assert adapter.resolutions() == ()

    def test_harvest_parses_close_time(self) -> None:
        adapter = LiveHarvestAdapter.harvest(
            [{"id": "a", "question": "q", "close_time": "2026-08-01"}],
            harvest_time=datetime(2026, 7, 1, tzinfo=UTC),
        )
        q = adapter.questions()[0]
        assert q.as_of.tzinfo is UTC
        assert q.close_time is not None and q.close_time.tzinfo is UTC
