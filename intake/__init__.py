"""Intake: question typing, normalization into resolvable objects, and refusal.

App layer (not shared ``core/``): DELPHI-specific question handling that turns an
arbitrary question into a normalized, resolvable object (or a principled refusal)
and opens its registry stream.
"""

from __future__ import annotations

from intake.llm import FixtureStructuredLLM, StructuredLLM
from intake.normalize import ResolvableQuestion, normalize_question
from intake.refusal import RefusalDecision, RefusalReason, assess_refusal
from intake.service import IntakeOutcome, IntakeService
from intake.typing import QuestionClassification, QuestionType, classify_question

__all__ = [
    "FixtureStructuredLLM",
    "IntakeOutcome",
    "IntakeService",
    "QuestionClassification",
    "QuestionType",
    "RefusalDecision",
    "RefusalReason",
    "ResolvableQuestion",
    "StructuredLLM",
    "assess_refusal",
    "classify_question",
    "normalize_question",
]
