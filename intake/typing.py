"""Question typing: classify an incoming question into a resolvable shape.

Step 1 of intake. The classifier is an LLM call behind the :class:`StructuredLLM`
seam; deterministic parsing maps its output into a typed classification, and any
unrecognized type collapses to ``UNKNOWN`` (which intake refuses downstream).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from intake.llm import StructuredLLM

__all__ = [
    "QuestionClassification",
    "QuestionType",
    "classify_question",
]


class QuestionType(StrEnum):
    """Forecast question shapes. Values mirror the registry ``QuestionType``.

    ``UNKNOWN`` is intake-only: it marks a question the classifier could not fit
    to a resolvable shape, and it never reaches the registry (it is refused).
    """

    BINARY = "binary"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"
    DATE = "date"
    UNKNOWN = "unknown"


class QuestionClassification(BaseModel):
    """The typed result of classifying a raw question."""

    model_config = ConfigDict(frozen=True)

    question_type: QuestionType
    entities: tuple[str, ...] = ()
    horizon: str | None = None


_CLASSIFY_SYSTEM = (
    "You classify forecasting questions. Return a JSON object with keys: "
    "question_type (one of 'binary', 'numeric', 'multiple_choice', 'date'), "
    "entities (array of the key named entities), and horizon (a short phrase for "
    "the time horizon, or null). If the text is not a well-posed question about a "
    "verifiable future or unresolved fact, set question_type to 'unknown'."
)


def _coerce_entities(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())


def _coerce_horizon(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def classify_question(question: str, *, llm: StructuredLLM) -> QuestionClassification:
    """Classify ``question`` into a :class:`QuestionClassification`.

    Raises ``ValueError`` for empty input. An unrecognized or missing type is
    mapped to :attr:`QuestionType.UNKNOWN` rather than raising, so the refusal
    stage owns the accept/reject decision.
    """
    if not question or not question.strip():
        msg = "question must be a non-empty string."
        raise ValueError(msg)
    raw = llm.invoke_structured(system=_CLASSIFY_SYSTEM, user=question)
    type_value = str(raw.get("question_type", "")).strip().lower()
    try:
        question_type = QuestionType(type_value)
    except ValueError:
        question_type = QuestionType.UNKNOWN
    return QuestionClassification(
        question_type=question_type,
        entities=_coerce_entities(raw.get("entities")),
        horizon=_coerce_horizon(raw.get("horizon")),
    )
