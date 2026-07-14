"""Tests for the OpenAI-compatible schema (C10.1)."""

from __future__ import annotations

import pytest

from api.compliance import usage_for
from api.envelope import ForecastEnvelope
from api.schema import ForecastAPIRequest, Message, build_response


def test_question_from_explicit_field() -> None:
    req = ForecastAPIRequest(question="Will X ship?", as_of="2024-06-01T00:00:00Z")
    assert req.resolved_question() == "Will X ship?"


def test_question_from_last_user_message() -> None:
    req = ForecastAPIRequest(
        messages=(
            Message(role="system", content="be careful"),
            Message(role="user", content="Will Y happen?"),
        ),
        as_of="2024-06-01T00:00:00Z",
    )
    assert req.resolved_question() == "Will Y happen?"


def test_question_skips_trailing_non_user_message() -> None:
    req = ForecastAPIRequest(
        messages=(
            Message(role="user", content="Will Z happen?"),
            Message(role="assistant", content="thinking..."),
        ),
        as_of="2024-06-01T00:00:00Z",
    )
    # The trailing assistant message is skipped; the earlier user message wins.
    assert req.resolved_question() == "Will Z happen?"


def test_missing_question_raises() -> None:
    req = ForecastAPIRequest(as_of="2024-06-01T00:00:00Z")
    with pytest.raises(ValueError, match="no question"):
        req.resolved_question()


def test_as_of_and_compliance() -> None:
    req = ForecastAPIRequest(
        question="q",
        as_of="2024-06-01T00:00:00Z",
        provider_opt_out=("openai",),
        retention_opt_out=True,
    )
    assert req.as_of_dt().year == 2024
    opts = req.compliance_options()
    assert opts.provider_opt_out == frozenset({"openai"})
    assert opts.retention_opt_out


def test_request_roundtrip_openai_shape() -> None:
    req = ForecastAPIRequest(question="q", as_of="2024-06-01T00:00:00Z")
    envelope = ForecastEnvelope(refused=False, probability=0.6, rationale="because")
    response = build_response(
        req, envelope, usage_for("delphi", model_calls=1), finish_reason="stop"
    )
    dumped = response.model_dump(mode="json")
    assert dumped["object"] == "forecast.completion"
    assert dumped["choices"][0]["message"]["content"] == "because"
    assert dumped["delphi"]["probability"] == pytest.approx(0.6)
    # created is deterministic (derived from as_of epoch).
    assert response.created == int(req.as_of_dt().timestamp())


def test_response_content_on_refusal() -> None:
    req = ForecastAPIRequest(question="q", as_of="2024-06-01T00:00:00Z")
    envelope = ForecastEnvelope(refused=True, refusal_reason="already_resolved")
    response = build_response(
        req, envelope, usage_for("delphi", model_calls=0), finish_reason="refusal"
    )
    assert "refused: already_resolved" in response.choices[0].message.content
