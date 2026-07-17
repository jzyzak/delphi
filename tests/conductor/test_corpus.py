"""Tests for the corpus tuple writer (C8.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from conductor.corpus import CorpusStore, CorpusWriter, FileCorpusStore, InMemoryCorpusStore
from conductor.heuristic import WorkflowTrace
from core.registry.models import ForecastInput, QuestionInput, ResolutionInput
from core.registry.store import InMemoryRegistryStore

_AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _workflow() -> WorkflowTrace:
    return WorkflowTrace(steps=(), route=("researcher",), revisions=0)


def _record_question(store: InMemoryRegistryStore) -> str:
    return store.record_question(
        QuestionInput(
            text="Will X ship?",
            question_type="binary",
            domain="tech",
            resolution_criteria="Resolves YES on GA.",
        )
    )


def _record_forecast(store: InMemoryRegistryStore, qid: str, *, probability: float = 0.8) -> str:
    return store.record_forecast(
        ForecastInput(
            question_id=qid,
            as_of=_AS_OF,
            probability=probability,
            rationale="r",
            model_provenance={"m": "v"},
            repro_handle={"as_of": _AS_OF.isoformat()},
        )
    )


def _record_resolution(store: InMemoryRegistryStore, qid: str, value: float) -> None:
    store.record_resolution(
        ResolutionInput(
            question_id=qid,
            resolved_value=value,
            resolved_at=datetime(2025, 1, 1, tzinfo=UTC),
            source="gov",
        )
    )


def test_in_memory_store_protocol_and_iteration() -> None:
    store = InMemoryCorpusStore()
    assert isinstance(store, CorpusStore)
    assert len(store) == 0


class TestCapture:
    def test_scored_when_resolved(self) -> None:
        store = InMemoryRegistryStore()
        corpus = InMemoryCorpusStore()
        qid = _record_question(store)
        _record_forecast(store, qid, probability=0.8)
        _record_resolution(store, qid, 1.0)
        writer = CorpusWriter(store=store, corpus=corpus)
        tuple_ = writer.capture(qid, _workflow())
        assert tuple_.resolved
        assert tuple_.proper_score == pytest.approx((0.8 - 1.0) ** 2)
        assert tuple_.scorer == "brier"
        assert len(corpus) == 1
        assert list(corpus)[0] is tuple_

    def test_unresolved_has_no_score(self) -> None:
        store = InMemoryRegistryStore()
        qid = _record_question(store)
        _record_forecast(store, qid)
        writer = CorpusWriter(store=store, corpus=InMemoryCorpusStore())
        tuple_ = writer.capture(qid, _workflow())
        assert not tuple_.resolved
        assert tuple_.proper_score is None

    def test_captures_latest_forecast(self) -> None:
        store = InMemoryRegistryStore()
        qid = _record_question(store)
        _record_forecast(store, qid, probability=0.3)
        _record_forecast(store, qid, probability=0.9)  # latest
        writer = CorpusWriter(store=store, corpus=InMemoryCorpusStore())
        tuple_ = writer.capture(qid, _workflow())
        assert tuple_.forecast.probability == pytest.approx(0.9)

    def test_non_binary_resolution_unscored(self) -> None:
        store = InMemoryRegistryStore()
        qid = _record_question(store)
        _record_forecast(store, qid)
        _record_resolution(store, qid, 0.5)  # non-binary outcome
        writer = CorpusWriter(store=store, corpus=InMemoryCorpusStore())
        tuple_ = writer.capture(qid, _workflow())
        assert tuple_.resolved
        assert tuple_.proper_score is None

    def test_no_forecast_raises(self) -> None:
        store = InMemoryRegistryStore()
        qid = _record_question(store)
        writer = CorpusWriter(store=store, corpus=InMemoryCorpusStore())
        with pytest.raises(ValueError, match="no forecast"):
            writer.capture(qid, _workflow())


class TestLatestFor:
    def test_returns_latest_and_none_for_unknown(self) -> None:
        store = InMemoryRegistryStore()
        corpus = InMemoryCorpusStore()
        qid = _record_question(store)
        _record_forecast(store, qid, probability=0.3)
        writer = CorpusWriter(store=store, corpus=corpus)
        first = writer.capture(qid, _workflow())
        _record_forecast(store, qid, probability=0.9)
        second = writer.capture(qid, _workflow())
        assert corpus.latest_for(qid) is second
        assert corpus.latest_for(qid) is not first
        assert corpus.latest_for("missing") is None


class TestRefresh:
    def test_completes_pending_tuple_after_resolution(self) -> None:
        store = InMemoryRegistryStore()
        corpus = InMemoryCorpusStore()
        qid = _record_question(store)
        _record_forecast(store, qid, probability=0.8)
        writer = CorpusWriter(store=store, corpus=corpus)
        pending = writer.capture(qid, _workflow())
        assert not pending.resolved

        _record_resolution(store, qid, 1.0)
        scored = writer.refresh(qid)
        assert scored is not None
        assert scored.resolved
        assert scored.proper_score == pytest.approx((0.8 - 1.0) ** 2)
        # The pending workflow trace was replayed, not lost.
        assert scored.workflow == pending.workflow
        assert len(corpus) == 2  # append-only: pending row + scored row

    def test_no_pending_row_is_a_noop(self) -> None:
        store = InMemoryRegistryStore()
        writer = CorpusWriter(store=store, corpus=InMemoryCorpusStore())
        assert writer.refresh("unknown") is None

    def test_still_unresolved_writes_nothing(self) -> None:
        store = InMemoryRegistryStore()
        corpus = InMemoryCorpusStore()
        qid = _record_question(store)
        _record_forecast(store, qid)
        writer = CorpusWriter(store=store, corpus=corpus)
        writer.capture(qid, _workflow())
        assert writer.refresh(qid) is None
        assert len(corpus) == 1

    def test_already_scored_returns_unchanged(self) -> None:
        store = InMemoryRegistryStore()
        corpus = InMemoryCorpusStore()
        qid = _record_question(store)
        _record_forecast(store, qid)
        _record_resolution(store, qid, 1.0)
        writer = CorpusWriter(store=store, corpus=corpus)
        scored = writer.capture(qid, _workflow())
        assert writer.refresh(qid) is scored
        assert len(corpus) == 1


class TestFileCorpusStore:
    def _scored_tuple(self, corpus: CorpusStore) -> None:
        store = InMemoryRegistryStore()
        qid = _record_question(store)
        _record_forecast(store, qid, probability=0.8)
        _record_resolution(store, qid, 1.0)
        CorpusWriter(store=store, corpus=corpus).capture(qid, _workflow())

    def test_round_trips_across_processes(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        first = FileCorpusStore(path)
        assert isinstance(first, CorpusStore)
        self._scored_tuple(first)
        assert len(first) == 1

        reloaded = FileCorpusStore(path)  # fresh process
        assert len(reloaded) == 1
        tuple_ = next(iter(reloaded))
        assert tuple_.resolved
        assert tuple_.proper_score == pytest.approx((0.8 - 1.0) ** 2)
        assert tuple_.workflow["route"] == ["researcher"]
        assert reloaded.latest_for(tuple_.question.question_id) is not None

    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        store = FileCorpusStore(tmp_path / "absent.jsonl")
        assert len(store) == 0
        assert store.latest_for("q") is None

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        store = FileCorpusStore(tmp_path / "deep" / "nested" / "corpus.jsonl")
        self._scored_tuple(store)
        assert len(FileCorpusStore(tmp_path / "deep" / "nested" / "corpus.jsonl")) == 1

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        store = FileCorpusStore(path)
        self._scored_tuple(store)
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(FileCorpusStore(path)) == 1

    def test_corrupt_line_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"not": "a corpus tuple"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="corrupt at line 1"):
            FileCorpusStore(path)

    def test_unparseable_json_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        path.write_text("{torn", encoding="utf-8")
        with pytest.raises(ValueError, match="corrupt at line 1"):
            FileCorpusStore(path)
