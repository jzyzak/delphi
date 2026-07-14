"""Tests for the benchmark resolution source (live-loop scoring)."""

from __future__ import annotations

from datetime import UTC, datetime

from benchmarks.base import BenchmarkResolution
from core.registry.models import Question
from resolution.benchmark_source import (
    BENCHMARK_QUESTION_ID_KEY,
    BenchmarkResolutionSource,
)

_RESOLVED_AT = datetime(2026, 6, 1, tzinfo=UTC)


def _question(metadata: dict[str, object]) -> Question:
    return Question(
        question_id="q-registry-1",
        text="Will X happen?",
        question_type="binary",
        domain="econ",
        resolution_criteria="official data",
        source="intake",
        metadata=metadata,
        knowledge_time=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _resolutions() -> list[BenchmarkResolution]:
    return [
        BenchmarkResolution(
            question_id="metaculus:101",
            resolved_value=1.0,
            resolved_at=_RESOLVED_AT,
            source="metaculus",
        )
    ]


class TestBenchmarkResolutionSource:
    def test_resolves_via_metadata_id(self) -> None:
        source = BenchmarkResolutionSource(_resolutions())
        question = _question({BENCHMARK_QUESTION_ID_KEY: "metaculus:101"})
        outcome = source.resolve(question)
        assert outcome is not None
        assert outcome.resolved_value == 1.0
        assert outcome.resolved_at == _RESOLVED_AT
        assert outcome.source == "metaculus"

    def test_unmatched_id_returns_none(self) -> None:
        source = BenchmarkResolutionSource(_resolutions())
        question = _question({BENCHMARK_QUESTION_ID_KEY: "metaculus:999"})
        assert source.resolve(question) is None

    def test_missing_metadata_returns_none(self) -> None:
        source = BenchmarkResolutionSource(_resolutions())
        assert source.resolve(_question({})) is None

    def test_non_string_metadata_returns_none(self) -> None:
        source = BenchmarkResolutionSource(_resolutions())
        assert source.resolve(_question({BENCHMARK_QUESTION_ID_KEY: 123})) is None

    def test_provenance_falls_back_to_question_sources(self) -> None:
        source = BenchmarkResolutionSource(
            [
                BenchmarkResolution(
                    question_id="metaculus:101",
                    resolved_value=0.0,
                    resolved_at=_RESOLVED_AT,
                )
            ]
        )
        question = _question(
            {
                BENCHMARK_QUESTION_ID_KEY: "metaculus:101",
                "resolution_sources": ["official site"],
            }
        )
        outcome = source.resolve(question)
        assert outcome is not None
        assert outcome.source == "official site"
