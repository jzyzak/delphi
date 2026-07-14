"""OpenAI-compatible request/response schema (C10.1).

The published surface mirrors an OpenAI chat-completions request so existing
clients point at DELPHI with no SDK migration (CLAUDE.md §9). DELPHI-specific
inputs (the as-of ceiling, the tier, and compliance opt-outs) ride as extra
fields. The question is taken from an explicit ``question`` field or the last
user message. The response is OpenAI-shaped with the full DELPHI envelope under
a ``delphi`` extension key.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from api.compliance import ComplianceOptions, UsageReport
from api.envelope import ForecastEnvelope
from benchmarks.base import parse_dt

__all__ = [
    "Choice",
    "ForecastAPIRequest",
    "ForecastAPIResponse",
    "Message",
    "build_response",
]

Tier = Literal["delphi", "delphi_deep"]


class Message(BaseModel):
    """One OpenAI-style chat message."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class ForecastAPIRequest(BaseModel):
    """OpenAI-compatible forecast request with DELPHI extensions."""

    model_config = ConfigDict(frozen=True)

    model: str = "delphi"
    messages: tuple[Message, ...] = ()
    question: str | None = None
    as_of: str
    tier: Tier = "delphi"
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

    def as_of_dt(self) -> datetime:
        """Parse the required as-of ceiling into a tz-aware UTC datetime."""
        return parse_dt(self.as_of)

    def compliance_options(self) -> ComplianceOptions:
        return ComplianceOptions(
            provider_opt_out=frozenset(self.provider_opt_out),
            retention_opt_out=self.retention_opt_out,
        )


class Choice(BaseModel):
    """One OpenAI-style completion choice."""

    model_config = ConfigDict(frozen=True)

    index: int
    message: Message
    finish_reason: str


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
