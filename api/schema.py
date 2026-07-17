"""OpenAI-compatible request/response schema (C10.1).

The published surface mirrors an OpenAI chat-completions request so existing
clients point at DELPHI with no SDK migration (CLAUDE.md §9). DELPHI-specific
inputs (the as-of ceiling, the tier, and compliance opt-outs) ride as extra
fields. The question is taken from an explicit ``question`` field or the last
user message. The forecast response is OpenAI-shaped with the full DELPHI
envelope under a ``delphi`` extension key.

The intake surfaces (``/v1/classify``, ``/v1/formalize``) share the same
question-bearing request shape but make ``as_of`` optional: intake is not
forecast-forming, and the ceiling only gates the "already resolved" refusal
check during formalization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.compliance import ComplianceOptions, UsageReport
from api.envelope import ForecastEnvelope
from api.jobs import ForecastJob
from benchmarks.base import parse_dt
from intake.service import IntakeOutcome
from intake.typing import QuestionClassification

__all__ = [
    "Choice",
    "ClassificationResult",
    "ClassifyAPIResponse",
    "ForecastAPIRequest",
    "ForecastAPIResponse",
    "ForecastJobAPIResponse",
    "ForecastJobSubmitRequest",
    "FormalizeAPIResponse",
    "FormalizedQuestion",
    "IntakeAPIRequest",
    "Message",
    "build_classify_response",
    "build_formalize_response",
    "build_job_response",
    "build_response",
]

Tier = Literal["delphi", "delphi_deep"]


class Message(BaseModel):
    """One OpenAI-style chat message."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class _QuestionAPIRequest(BaseModel):
    """Shared question-bearing request fields (OpenAI-message compatible)."""

    model_config = ConfigDict(frozen=True)

    model: str = "delphi"
    messages: tuple[Message, ...] = ()
    question: str | None = None
    provider_opt_out: tuple[str, ...] = ()
    retention_opt_out: bool = False

    def resolved_question(self) -> str:
        """The question text: explicit ``question`` or the last user message."""
        if self.question and self.question.strip():
            return self.question.strip()
        for message in reversed(self.messages):
            if message.role == "user" and message.content.strip():
                return message.content.strip()
        msg = "no question provided (set 'question' or include a user message)."
        raise ValueError(msg)

    def compliance_options(self) -> ComplianceOptions:
        return ComplianceOptions(
            provider_opt_out=frozenset(self.provider_opt_out),
            retention_opt_out=self.retention_opt_out,
        )


class ForecastAPIRequest(_QuestionAPIRequest):
    """OpenAI-compatible forecast request with DELPHI extensions."""

    as_of: str
    tier: Tier = "delphi"

    def as_of_dt(self) -> datetime:
        """Parse the required as-of ceiling into a tz-aware UTC datetime."""
        return parse_dt(self.as_of)


class ForecastJobSubmitRequest(ForecastAPIRequest):
    """``POST /v1/forecast/jobs``: a forecast request + optional idempotency key.

    The key lets a client retry the submit safely: one key maps to exactly one
    job, so a retried POST returns the existing job instead of paying for a
    second forecast.
    """

    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    def job_payload(self) -> dict[str, Any]:
        """The stored request payload (the forecast request, sans the key)."""
        return self.model_dump(mode="json", exclude={"idempotency_key"})


class IntakeAPIRequest(_QuestionAPIRequest):
    """Request for the intake surfaces (``/v1/classify``, ``/v1/formalize``)."""

    as_of: str | None = None

    def as_of_dt(self) -> datetime | None:
        """Parse the optional as-of ceiling, or return ``None`` when absent."""
        if self.as_of is None or not self.as_of.strip():
            return None
        return parse_dt(self.as_of)


class Choice(BaseModel):
    """One OpenAI-style completion choice."""

    model_config = ConfigDict(frozen=True)

    index: int
    message: Message
    finish_reason: str


class ClassificationResult(BaseModel):
    """The typed classification of a question (mirrors intake typing)."""

    model_config = ConfigDict(frozen=True)

    question_type: str
    entities: tuple[str, ...] = ()
    horizon: str | None = None

    @classmethod
    def from_classification(cls, classification: QuestionClassification) -> ClassificationResult:
        return cls(
            question_type=classification.question_type.value,
            entities=classification.entities,
            horizon=classification.horizon,
        )


class ClassifyAPIResponse(BaseModel):
    """Response for ``/v1/classify``."""

    model_config = ConfigDict(frozen=True)

    id: str = "cl-delphi"
    object: str = "question.classification"
    model: str
    classification: ClassificationResult
    usage: UsageReport


class FormalizedQuestion(BaseModel):
    """The normalized, resolvable form of a question (pre-registry)."""

    model_config = ConfigDict(frozen=True)

    text: str
    question_type: str
    domain: str
    resolution_criteria: str
    resolution_sources: tuple[str, ...] = ()
    close_time: datetime | None = None
    entities: tuple[str, ...] = ()


class FormalizeAPIResponse(BaseModel):
    """Response for ``/v1/formalize`` — the resolvable form, or a refusal."""

    model_config = ConfigDict(frozen=True)

    id: str = "fm-delphi"
    object: str = "question.formalization"
    model: str
    refused: bool
    refusal_reason: str = ""
    refusal_detail: str = ""
    classification: ClassificationResult
    formalized: FormalizedQuestion | None = None
    usage: UsageReport


class ForecastAPIResponse(BaseModel):
    """OpenAI-compatible response carrying the DELPHI envelope."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: str = "forecast.completion"
    created: int
    model: str
    choices: tuple[Choice, ...]
    usage: UsageReport
    delphi: ForecastEnvelope


class ForecastJobAPIResponse(BaseModel):
    """The job resource returned by the submit and status endpoints.

    ``result`` is the full ``ForecastAPIResponse`` payload once the job
    succeeds (kept as the stored JSON rather than re-validated — it was built
    by :func:`build_response` at execution time).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    object: str = "forecast.job"
    status: str
    created: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


def build_response(
    request: ForecastAPIRequest,
    envelope: ForecastEnvelope,
    usage: UsageReport,
    *,
    finish_reason: str,
) -> ForecastAPIResponse:
    """Assemble the OpenAI-shaped response around the DELPHI envelope."""
    content = envelope.rationale if not envelope.refused else f"refused: {envelope.refusal_reason}"
    return ForecastAPIResponse(
        id=f"fc-{request.tier}",
        created=int(request.as_of_dt().timestamp()),
        model=request.model,
        choices=(
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason=finish_reason,
            ),
        ),
        usage=usage,
        delphi=envelope,
    )


def build_job_response(job: ForecastJob) -> ForecastJobAPIResponse:
    """Assemble the job resource from a persisted job row."""
    created = int(job.created_at.timestamp()) if job.created_at is not None else 0
    return ForecastJobAPIResponse(
        id=job.job_id,
        status=job.status.value,
        created=created,
        started_at=job.started_at,
        finished_at=job.finished_at,
        request=dict(job.request),
        result=None if job.result is None else dict(job.result),
        error=job.error,
    )


def build_classify_response(
    request: IntakeAPIRequest,
    classification: QuestionClassification,
    usage: UsageReport,
) -> ClassifyAPIResponse:
    """Assemble the ``/v1/classify`` response from an intake classification."""
    return ClassifyAPIResponse(
        model=request.model,
        classification=ClassificationResult.from_classification(classification),
        usage=usage,
    )


def build_formalize_response(
    request: IntakeAPIRequest,
    outcome: IntakeOutcome,
    usage: UsageReport,
) -> FormalizeAPIResponse:
    """Assemble the ``/v1/formalize`` response from a (non-recording) intake outcome."""
    formalized = None
    if outcome.accepted and outcome.resolvable is not None:
        resolvable = outcome.resolvable
        formalized = FormalizedQuestion(
            text=resolvable.text,
            question_type=resolvable.question_type.value,
            domain=resolvable.domain,
            resolution_criteria=resolvable.resolution_criteria,
            resolution_sources=resolvable.resolution_sources,
            close_time=resolvable.close_time,
            entities=resolvable.entities,
        )
    refusal = outcome.refusal
    return FormalizeAPIResponse(
        model=request.model,
        refused=not outcome.accepted,
        refusal_reason=(refusal.reason.value if refusal and refusal.reason else ""),
        refusal_detail=(refusal.detail if refusal else ""),
        classification=ClassificationResult.from_classification(outcome.classification),
        formalized=formalized,
        usage=usage,
    )
