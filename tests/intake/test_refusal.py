"""Unit tests for the deterministic refusal gate (intake step 3)."""

from __future__ import annotations

from datetime import UTC, datetime

from intake.normalize import ResolvableQuestion
from intake.refusal import RefusalReason, assess_refusal
from intake.typing import QuestionClassification, QuestionType

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _classification(qt: QuestionType = QuestionType.BINARY) -> QuestionClassification:
    return QuestionClassification(question_type=qt)


def _resolvable(**overrides: object) -> ResolvableQuestion:
    base: dict[str, object] = {
        "text": "Will X ship before 2025?",
        "question_type": QuestionType.BINARY,
        "domain": "tech",
        "resolution_criteria": "Resolves YES if GA before 2025-01-01.",
        "close_time": datetime(2025, 1, 1, tzinfo=UTC),
        "resolvable": True,
    }
    base.update(overrides)
    return ResolvableQuestion(**base)  # type: ignore[arg-type]


class TestAssessRefusal:
    def test_unknown_type_refused(self) -> None:
        decision = assess_refusal(_classification(QuestionType.UNKNOWN), None)
        assert decision.refused
        assert decision.reason is RefusalReason.UNKNOWN_TYPE

    def test_missing_resolvable_refused(self) -> None:
        decision = assess_refusal(_classification(), None)
        assert decision.reason is RefusalReason.UNDERSPECIFIED

    def test_unresolvable_scope_hint_maps_out_of_scope(self) -> None:
        resolvable = _resolvable(resolvable=False, refusal_hint="subjective opinion")
        decision = assess_refusal(_classification(), resolvable)
        assert decision.reason is RefusalReason.OUT_OF_SCOPE
        assert decision.detail == "subjective opinion"

    def test_unresolvable_other_hint_maps_unresolvable(self) -> None:
        resolvable = _resolvable(resolvable=False, refusal_hint="no data source exists")
        assert assess_refusal(_classification(), resolvable).reason is RefusalReason.UNRESOLVABLE

    def test_unresolvable_empty_hint_uses_default_detail(self) -> None:
        resolvable = _resolvable(resolvable=False, refusal_hint="")
        decision = assess_refusal(_classification(), resolvable)
        assert decision.reason is RefusalReason.UNRESOLVABLE
        assert decision.detail == "Question reported as not resolvable."

    def test_empty_criteria_refused(self) -> None:
        decision = assess_refusal(_classification(), _resolvable(resolution_criteria="  "))
        assert decision.reason is RefusalReason.UNDERSPECIFIED

    def test_already_resolved_when_close_before_as_of(self) -> None:
        resolvable = _resolvable(close_time=datetime(2024, 1, 1, tzinfo=UTC))
        decision = assess_refusal(_classification(), resolvable, as_of=AS_OF)
        assert decision.reason is RefusalReason.ALREADY_RESOLVED

    def test_future_close_not_refused(self) -> None:
        decision = assess_refusal(_classification(), _resolvable(), as_of=AS_OF)
        assert not decision.refused

    def test_no_as_of_skips_time_check(self) -> None:
        resolvable = _resolvable(close_time=datetime(2024, 1, 1, tzinfo=UTC))
        assert not assess_refusal(_classification(), resolvable).refused

    def test_accepted(self) -> None:
        decision = assess_refusal(_classification(), _resolvable())
        assert not decision.refused
        assert decision.reason is None
