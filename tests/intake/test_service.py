"""Unit tests for the intake service (classify -> normalize -> refuse-or-record)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.registry.store import InMemoryRegistryStore
from intake.llm import FixtureStructuredLLM
from intake.refusal import RefusalReason
from intake.service import IntakeService

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)

_CLASSIFY_BINARY = {"question_type": "binary", "entities": ["X"], "horizon": "2025"}
_NORMALIZE_OK = {
    "canonical_text": "Will X reach GA before 2025-01-01?",
    "domain": "tech",
    "resolution_criteria": "Resolves YES if GA is announced before 2025-01-01.",
    "resolution_sources": ["vendor blog"],
    "close_time": "2025-01-01T00:00:00+00:00",
    "resolvable": True,
}


def _service(*responses: dict[str, object]) -> tuple[IntakeService, InMemoryRegistryStore]:
    store = InMemoryRegistryStore()
    service = IntakeService(llm=FixtureStructuredLLM(list(responses)), store=store)
    return service, store


class TestIntakeService:
    def test_accepted_records_question(self) -> None:
        service, store = _service(_CLASSIFY_BINARY, _NORMALIZE_OK)
        outcome = service.intake("Will X ship?")
        assert outcome.accepted
        assert outcome.question_id is not None
        assert outcome.refusal is None
        recorded = store.get_question(outcome.question_id)
        assert recorded.domain == "tech"
        assert recorded.question_type == "binary"
        assert recorded.source == "intake"
        assert recorded.metadata["entities"] == ["X"]
        assert recorded.metadata["resolution_sources"] == ["vendor blog"]

    def test_unknown_type_refused_without_recording(self) -> None:
        service, store = _service({"question_type": "unknown"})
        outcome = service.intake("Mmm?")
        assert not outcome.accepted
        assert outcome.question_id is None
        assert outcome.refusal is not None
        assert outcome.refusal.reason is RefusalReason.UNKNOWN_TYPE
        assert store.questions_by_domain("tech") == ()

    def test_refused_after_normalize_not_recorded(self) -> None:
        service, store = _service(_CLASSIFY_BINARY, {"resolution_criteria": ""})
        outcome = service.intake("Will X ship?")
        assert not outcome.accepted
        assert outcome.resolvable is not None
        assert outcome.refusal is not None
        assert outcome.refusal.reason is RefusalReason.UNDERSPECIFIED
        assert store.questions_by_domain("general") == ()

    def test_already_resolved_with_as_of(self) -> None:
        normalize_past = {**_NORMALIZE_OK, "close_time": "2024-01-01T00:00:00+00:00"}
        service, _store = _service(_CLASSIFY_BINARY, normalize_past)
        outcome = service.intake("Did X ship?", as_of=AS_OF)
        assert not outcome.accepted
        assert outcome.refusal is not None
        assert outcome.refusal.reason is RefusalReason.ALREADY_RESOLVED

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_question_raises(self, text: str) -> None:
        service, _store = _service()
        with pytest.raises(ValueError, match="non-empty"):
            service.intake(text)


class TestClassify:
    def test_returns_classification(self) -> None:
        service, _store = _service(_CLASSIFY_BINARY)
        classification = service.classify("Will X ship?")
        assert classification.question_type.value == "binary"
        assert classification.entities == ("X",)
        assert classification.horizon == "2025"

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_question_raises(self, text: str) -> None:
        service, _store = _service()
        with pytest.raises(ValueError, match="non-empty"):
            service.classify(text)

    def test_does_not_record(self) -> None:
        service, store = _service(_CLASSIFY_BINARY)
        service.classify("Will X ship?")
        assert store.questions_by_domain("tech") == ()


class TestAssess:
    def test_accepted_without_recording(self) -> None:
        service, store = _service(_CLASSIFY_BINARY, _NORMALIZE_OK)
        outcome = service.assess("Will X ship?")
        assert outcome.accepted
        assert outcome.question_id is None  # assess never records
        assert outcome.resolvable is not None
        assert outcome.resolvable.domain == "tech"
        assert store.questions_by_domain("tech") == ()

    def test_unknown_type_refused(self) -> None:
        service, store = _service({"question_type": "unknown"})
        outcome = service.assess("Mmm?")
        assert not outcome.accepted
        assert outcome.refusal is not None
        assert outcome.refusal.reason is RefusalReason.UNKNOWN_TYPE
        assert store.questions_by_domain("general") == ()

    def test_underspecified_refused(self) -> None:
        service, _store = _service(_CLASSIFY_BINARY, {"resolution_criteria": ""})
        outcome = service.assess("Will X ship?")
        assert not outcome.accepted
        assert outcome.refusal is not None
        assert outcome.refusal.reason is RefusalReason.UNDERSPECIFIED

    def test_already_resolved_with_as_of(self) -> None:
        normalize_past = {**_NORMALIZE_OK, "close_time": "2024-01-01T00:00:00+00:00"}
        service, _store = _service(_CLASSIFY_BINARY, normalize_past)
        outcome = service.assess("Did X ship?", as_of=AS_OF)
        assert not outcome.accepted
        assert outcome.refusal is not None
        assert outcome.refusal.reason is RefusalReason.ALREADY_RESOLVED

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_question_raises(self, text: str) -> None:
        service, _store = _service()
        with pytest.raises(ValueError, match="non-empty"):
            service.assess(text)
