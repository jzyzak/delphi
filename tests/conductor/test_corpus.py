"""Tests for the corpus tuple writer (C8.3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from conductor.corpus import CorpusStore, CorpusWriter, InMemoryCorpusStore
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
