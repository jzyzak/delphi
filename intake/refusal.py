"""Refusal policy: refuse unresolvable questions instead of guessing.

Step 3 of intake. Answering an unresolvable question is punditry, not
forecasting (CLAUDE.md §10). This stage is deterministic: it enforces a gate over
the structured outputs of typing and normalization, and never calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intake.normalize import ResolvableQuestion
from intake.typing import QuestionClassification, QuestionType

__all__ = [
    "RefusalDecision",
    "RefusalReason",
    "assess_refusal",
]


class RefusalReason(StrEnum):
    """Why a question was refused at intake."""

    UNKNOWN_TYPE = "unknown_type"
    UNDERSPECIFIED = "underspecified"
    UNRESOLVABLE = "unresolvable"
    OUT_OF_SCOPE = "out_of_scope"
    ALREADY_RESOLVED = "already_resolved"


@dataclass(frozen=True)
class RefusalDecision:
    """The outcome of the refusal gate."""

    refused: bool
    reason: RefusalReason | None = None
    detail: str = ""


_OUT_OF_SCOPE_MARKERS = ("scope", "opinion", "subjective", "normative", "value judgment")


def _map_hint(hint: str) -> RefusalReason:
    lowered = hint.lower()
    if any(marker in lowered for marker in _OUT_OF_SCOPE_MARKERS):
        return RefusalReason.OUT_OF_SCOPE
    return RefusalReason.UNRESOLVABLE


def assess_refusal(
    classification: QuestionClassification,
    resolvable: ResolvableQuestion | None,
    *,
    as_of: datetime | None = None,
) -> RefusalDecision:
    """Decide whether to refuse a question.

    Order of checks is deliberate: an unknown type short-circuits before
    normalization is even attempted; a missing normalized form, an explicit
    unresolvable flag, absent criteria, and a past close time follow. ``as_of``
    is optional and only gates the "already resolved" check when provided.
    """
    if classification.question_type is QuestionType.UNKNOWN:
        return RefusalDecision(
            True, RefusalReason.UNKNOWN_TYPE, "Question type could not be determined."
        )
    if resolvable is None:
        return RefusalDecision(
            True, RefusalReason.UNDERSPECIFIED, "No normalized resolvable form was produced."
        )
    if not resolvable.resolvable:
        detail = resolvable.refusal_hint or "Question reported as not resolvable."
        return RefusalDecision(True, _map_hint(resolvable.refusal_hint), detail)
    if not resolvable.resolution_criteria.strip():
        return RefusalDecision(
            True, RefusalReason.UNDERSPECIFIED, "No resolution criteria could be established."
        )
    if as_of is not None and resolvable.close_time is not None and resolvable.close_time <= as_of:
        return RefusalDecision(
            True,
            RefusalReason.ALREADY_RESOLVED,
            "Close time is at or before the as-of time.",
        )
    return RefusalDecision(False)
