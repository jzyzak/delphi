"""Intake service: classify -> normalize -> refuse-or-record.

Composes the intake stages and, on acceptance, writes the immutable ``question``
genesis record to the registry (opening its forecast stream). Refused questions
are returned to the caller and not recorded, so the registry holds only
resolvable questions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.registry.models import QuestionInput
from core.registry.store import RegistryStore
from intake.llm import StructuredLLM
from intake.normalize import ResolvableQuestion, normalize_question
from intake.refusal import RefusalDecision, assess_refusal
from intake.typing import QuestionClassification, QuestionType, classify_question

__all__ = [
    "IntakeOutcome",
    "IntakeService",
]


@dataclass(frozen=True)
class IntakeOutcome:
    """The result of running a question through intake."""

    accepted: bool
    classification: QuestionClassification
    resolvable: ResolvableQuestion | None
    refusal: RefusalDecision | None
    question_id: str | None


def _to_question_input(
    resolvable: ResolvableQuestion, *, extra_metadata: Mapping[str, Any] | None = None
) -> QuestionInput:
    metadata: dict[str, Any] = {
        "entities": list(resolvable.entities),
        "resolution_sources": list(resolvable.resolution_sources),
    }
    if extra_metadata:
        # Caller-supplied provenance (e.g. the benchmark question id threaded from
        # a live harvest) so resolution can map this question back to its source.
        metadata.update(extra_metadata)
    return QuestionInput(
        text=resolvable.text,
        question_type=resolvable.question_type.value,  # type: ignore[arg-type]
        domain=resolvable.domain,
        resolution_criteria=resolvable.resolution_criteria,
        close_time=resolvable.close_time,
        source="intake",
        metadata=metadata,
    )


class IntakeService:
    """Runs intake and records accepted questions to the registry."""

    def __init__(self, *, llm: StructuredLLM, store: RegistryStore) -> None:
        self._llm = llm
        self._store = store

    def intake(
        self,
        question_text: str,
        *,
        as_of: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> IntakeOutcome:
        """Classify, normalize, and either refuse or record ``question_text``.

        ``as_of`` is optional; when provided it enables the "already resolved"
        refusal check. Intake is not forecast-forming, so it never reads the world
        through the as-of facade. ``metadata`` is merged into the recorded
        question's metadata (e.g. a benchmark question id for later resolution).
        """
        if not question_text or not question_text.strip():
            msg = "question_text must be a non-empty string."
            raise ValueError(msg)

        classification = classify_question(question_text, llm=self._llm)
        if classification.question_type is QuestionType.UNKNOWN:
            decision = assess_refusal(classification, None, as_of=as_of)
            return IntakeOutcome(False, classification, None, decision, None)

        resolvable = normalize_question(question_text, classification, llm=self._llm)
        decision = assess_refusal(classification, resolvable, as_of=as_of)
        if decision.refused:
            return IntakeOutcome(False, classification, resolvable, decision, None)

        question_id = self._store.record_question(
            _to_question_input(resolvable, extra_metadata=metadata)
        )
        return IntakeOutcome(True, classification, resolvable, None, question_id)
