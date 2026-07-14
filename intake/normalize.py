"""Normalization: turn a classified question into a resolvable object.

Step 2 of intake. Produces the canonical text, an explicit machine-checkable
resolution criterion, a domain (for per-domain calibration, §2.3), resolution
source hints, and a close time. The LLM also reports whether the question is
resolvable at all; the refusal stage enforces the gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from intake.llm import StructuredLLM
from intake.typing import QuestionClassification, QuestionType

__all__ = [
    "ResolvableQuestion",
    "normalize_question",
]


class ResolvableQuestion(BaseModel):
    """A question normalized into a resolvable form (pre-registry)."""

    model_config = ConfigDict(frozen=True)

    text: str
    question_type: QuestionType
    domain: str
    resolution_criteria: str
    resolution_sources: tuple[str, ...] = ()
    close_time: datetime | None = None
    entities: tuple[str, ...] = ()
    resolvable: bool = True
    refusal_hint: str = ""


_NORMALIZE_SYSTEM = (
    "You normalize a forecasting question into a resolvable form. Return a JSON "
    "object with keys: canonical_text (a precise restatement), domain (a short "
    "topic label like 'geopolitics' or 'tech'), resolution_criteria (an explicit, "
    "verifiable rule for how it resolves), resolution_sources (array of where the "
    "outcome can be checked), close_time (ISO-8601 timestamp when it can be "
    "resolved, or null), resolvable (boolean), and refusal_reason (short reason if "
    "not resolvable, else empty). Do not invent facts; only structure the question."
)


def _parse_close_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _coerce_str(raw: Any, default: str = "") -> str:
    return raw.strip() if isinstance(raw, str) and raw.strip() else default


def _coerce_sources(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())


def normalize_question(
    question: str,
    classification: QuestionClassification,
    *,
    llm: StructuredLLM,
) -> ResolvableQuestion:
    """Normalize ``question`` into a :class:`ResolvableQuestion`.

    Fields the model omits fall back to safe defaults (canonical text -> original
    question, domain -> ``"general"``, criteria -> empty which the refusal stage
    treats as underspecified). ``resolvable`` defaults to True unless the model
    explicitly reports otherwise.
    """
    raw = llm.invoke_structured(system=_NORMALIZE_SYSTEM, user=question)
    return ResolvableQuestion(
        text=_coerce_str(raw.get("canonical_text"), question.strip()),
        question_type=classification.question_type,
        domain=_coerce_str(raw.get("domain"), "general"),
        resolution_criteria=_coerce_str(raw.get("resolution_criteria")),
        resolution_sources=_coerce_sources(raw.get("resolution_sources")),
        close_time=_parse_close_time(raw.get("close_time")),
        entities=classification.entities,
        resolvable=bool(raw.get("resolvable", True)),
        refusal_hint=_coerce_str(raw.get("refusal_reason")),
    )
