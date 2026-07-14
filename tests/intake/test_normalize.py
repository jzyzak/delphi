"""Unit tests for normalization (intake step 2). Hermetic, mocked LLM."""

from __future__ import annotations

from datetime import UTC, datetime

from intake.llm import FixtureStructuredLLM
from intake.normalize import normalize_question
from intake.typing import QuestionClassification, QuestionType


def _classification(**overrides: object) -> QuestionClassification:
    base: dict[str, object] = {
        "question_type": QuestionType.BINARY,
        "entities": ("X",),
        "horizon": "2025",
    }
    base.update(overrides)
    return QuestionClassification(**base)  # type: ignore[arg-type]


class TestNormalizeQuestion:
    def test_full_fields(self) -> None:
        llm = FixtureStructuredLLM(
            {
                "canonical_text": "Will X reach GA before 2025-01-01?",
                "domain": "tech",
                "resolution_criteria": "Resolves YES if GA is announced before 2025-01-01.",
                "resolution_sources": ["vendor blog", "press release"],
                "close_time": "2025-01-01T00:00:00+00:00",
                "resolvable": True,
            }
        )
        result = normalize_question("Will X ship?", _classification(), llm=llm)
        assert result.text == "Will X reach GA before 2025-01-01?"
        assert result.domain == "tech"
        assert result.resolution_sources == ("vendor blog", "press release")
        assert result.close_time == datetime(2025, 1, 1, tzinfo=UTC)
        assert result.question_type is QuestionType.BINARY
        assert result.entities == ("X",)
        assert result.resolvable is True

    def test_defaults_when_missing(self) -> None:
        result = normalize_question(
            "Raw question?", _classification(), llm=FixtureStructuredLLM({})
        )
        assert result.text == "Raw question?"
        assert result.domain == "general"
        assert result.resolution_criteria == ""
        assert result.resolution_sources == ()
        assert result.close_time is None
        assert result.resolvable is True
        assert result.refusal_hint == ""

    def test_naive_close_time_assumed_utc(self) -> None:
        llm = FixtureStructuredLLM({"close_time": "2025-01-01T00:00:00"})
        result = normalize_question("q?", _classification(), llm=llm)
        assert result.close_time == datetime(2025, 1, 1, tzinfo=UTC)

    def test_invalid_close_time_becomes_none(self) -> None:
        llm = FixtureStructuredLLM({"close_time": "not-a-date"})
        assert normalize_question("q?", _classification(), llm=llm).close_time is None

    def test_non_string_close_time_becomes_none(self) -> None:
        llm = FixtureStructuredLLM({"close_time": 2025})
        assert normalize_question("q?", _classification(), llm=llm).close_time is None

    def test_non_list_sources_become_empty(self) -> None:
        llm = FixtureStructuredLLM({"resolution_sources": "a blog"})
        assert normalize_question("q?", _classification(), llm=llm).resolution_sources == ()

    def test_unresolvable_with_hint(self) -> None:
        llm = FixtureStructuredLLM(
            {"resolvable": False, "refusal_reason": "opinion question, out of scope"}
        )
        result = normalize_question("q?", _classification(), llm=llm)
        assert result.resolvable is False
        assert result.refusal_hint == "opinion question, out of scope"
