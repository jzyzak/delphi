"""Unit tests for question typing (intake step 1). Hermetic, mocked LLM."""

from __future__ import annotations

import pytest

from intake.llm import FixtureStructuredLLM
from intake.typing import QuestionType, classify_question


class TestClassifyQuestion:
    @pytest.mark.parametrize(
        "value",
        ["binary", "numeric", "multiple_choice", "date"],
    )
    def test_recognized_types(self, value: str) -> None:
        llm = FixtureStructuredLLM({"question_type": value})
        result = classify_question("Some question?", llm=llm)
        assert result.question_type == QuestionType(value)
        assert llm.calls  # the seam was actually used

    def test_uppercase_type_is_normalized(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "BINARY"})
        assert classify_question("q?", llm=llm).question_type is QuestionType.BINARY

    def test_missing_type_maps_to_unknown(self) -> None:
        llm = FixtureStructuredLLM({})
        assert classify_question("q?", llm=llm).question_type is QuestionType.UNKNOWN

    def test_invalid_type_maps_to_unknown(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "vibes"})
        assert classify_question("q?", llm=llm).question_type is QuestionType.UNKNOWN

    def test_entities_coerced(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "binary", "entities": ["X", " ", 7, "Y "]})
        assert classify_question("q?", llm=llm).entities == ("X", "Y")

    def test_entities_non_list_becomes_empty(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "binary", "entities": "X"})
        assert classify_question("q?", llm=llm).entities == ()

    def test_horizon_coerced(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "date", "horizon": " 2025 "})
        assert classify_question("q?", llm=llm).horizon == "2025"

    def test_horizon_non_string_becomes_none(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "date", "horizon": 2025})
        assert classify_question("q?", llm=llm).horizon is None

    def test_blank_horizon_becomes_none(self) -> None:
        llm = FixtureStructuredLLM({"question_type": "date", "horizon": "  "})
        assert classify_question("q?", llm=llm).horizon is None

    @pytest.mark.parametrize("question", ["", "   "])
    def test_empty_question_raises(self, question: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            classify_question(question, llm=FixtureStructuredLLM())


class TestFixtureStructuredLLM:
    def test_queue_pops_in_order_then_empty(self) -> None:
        llm = FixtureStructuredLLM([{"a": 1}, {"b": 2}])
        assert llm.invoke_structured(system="s", user="u") == {"a": 1}
        assert llm.invoke_structured(system="s", user="u") == {"b": 2}
        assert llm.invoke_structured(system="s", user="u") == {}
        assert len(llm.calls) == 3

    def test_none_responses_returns_empty(self) -> None:
        assert FixtureStructuredLLM().invoke_structured(system="s", user="u") == {}
