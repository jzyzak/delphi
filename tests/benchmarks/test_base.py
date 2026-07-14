"""Tests for the benchmark adapter interface + harness feed (C7.1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkQuestion,
    BenchmarkResolution,
    as_of_pins,
    assert_no_leakage,
    parse_dt,
    scored_records,
)

_AS_OF = datetime(2026, 1, 1, tzinfo=UTC)
_RESOLVED = datetime(2026, 6, 1, tzinfo=UTC)


class _Adapter:
    def __init__(
        self,
        questions: tuple[BenchmarkQuestion, ...],
        resolutions: tuple[BenchmarkResolution, ...],
    ) -> None:
        self._q = questions
        self._r = resolutions

    @property
    def name(self) -> str:
        return "fixture"

    def questions(self) -> tuple[BenchmarkQuestion, ...]:
        return self._q

    def resolutions(self) -> tuple[BenchmarkResolution, ...]:
        return self._r


def _question(external_id: str, *, domain: str = "econ") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        source="fixture", external_id=external_id, text="q?", as_of=_AS_OF, domain=domain
    )


def _resolution(qid: str, value: float, *, at: datetime = _RESOLVED) -> BenchmarkResolution:
    return BenchmarkResolution(question_id=qid, resolved_value=value, resolved_at=at)


def test_question_id_and_utc_conversion() -> None:
    q = BenchmarkQuestion(
        source="s",
        external_id="1",
        text="q",
        as_of=datetime(2026, 1, 1, 5, tzinfo=timezone(timedelta(hours=5))),
        close_time=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert q.question_id == "s:1"
    assert q.as_of == datetime(2026, 1, 1, tzinfo=UTC)  # converted to UTC
    assert q.close_time is not None and q.close_time.tzinfo is UTC


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="[Nn]aive"):
        BenchmarkQuestion(source="s", external_id="1", text="q", as_of=datetime(2026, 1, 1))


def test_protocol_conformance() -> None:
    adapter = _Adapter((_question("1"),), ())
    assert isinstance(adapter, BenchmarkAdapter)


def test_as_of_pins() -> None:
    adapter = _Adapter((_question("1"), _question("2")), ())
    assert as_of_pins(adapter) == {"fixture:1": _AS_OF, "fixture:2": _AS_OF}


class TestScoredRecords:
    def test_joins_forecasts_to_resolutions(self) -> None:
        adapter = _Adapter(
            (_question("1", domain="econ"), _question("2", domain="geo")),
            (_resolution("fixture:1", 1.0), _resolution("fixture:2", 0.0)),
        )
        records = scored_records({"fixture:1": 0.8, "fixture:2": 0.2}, adapter)
        assert {r.question_id for r in records} == {"fixture:1", "fixture:2"}
        assert {r.domain for r in records} == {"econ", "geo"}

    def test_skips_unforecast_questions(self) -> None:
        adapter = _Adapter((_question("1"),), (_resolution("fixture:1", 1.0),))
        assert scored_records({}, adapter) == ()

    def test_skips_non_binary_outcomes(self) -> None:
        adapter = _Adapter((_question("1"),), (_resolution("fixture:1", 0.5),))
        assert scored_records({"fixture:1": 0.8}, adapter) == ()


class TestNoLeakage:
    def test_passes_when_resolved_after_pin(self) -> None:
        assert_no_leakage([_question("1")], [_resolution("fixture:1", 1.0)])

    def test_raises_when_resolved_before_pin(self) -> None:
        early = datetime(2025, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage([_question("1")], [_resolution("fixture:1", 1.0, at=early)])

    def test_ignores_resolution_without_question(self) -> None:
        assert_no_leakage([_question("1")], [_resolution("fixture:999", 1.0)])


class TestParseDt:
    def test_parses_z_suffix_and_naive(self) -> None:
        assert parse_dt("2026-01-01T00:00:00Z") == _AS_OF
        assert parse_dt("2026-01-01").tzinfo is UTC

    def test_passthrough_datetime(self) -> None:
        assert parse_dt(_AS_OF) == _AS_OF

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="ISO-8601"):
            parse_dt("")
