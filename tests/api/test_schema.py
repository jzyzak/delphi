"""Tests for the OpenAI-compatible schema (C10.1)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from api.compliance import usage_for
from api.envelope import ForecastEnvelope
from api.jobs import ForecastJob, JobStatus
from api.schema import (
    ForecastAPIRequest,
    ForecastJobSubmitRequest,
    IntakeAPIRequest,
    Message,
    build_classify_response,
    build_formalize_response,
    build_job_response,
    build_response,
)
from intake.normalize import ResolvableQuestion
from intake.refusal import RefusalDecision, RefusalReason
from intake.service import IntakeOutcome
from intake.typing import QuestionClassification, QuestionType

_CLASSIFICATION = QuestionClassification(
    question_type=QuestionType.BINARY, entities=("X",), horizon="2025"
)
_RESOLVABLE = ResolvableQuestion(
    text="Will X reach GA before 2025-01-01?",
    question_type=QuestionType.BINARY,
    domain="tech",
    resolution_criteria="Resolves YES if GA is announced before 2025-01-01.",
    resolution_sources=("vendor blog",),
    entities=("X",),
)


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


class TestIntakeAPIRequest:
    def test_as_of_is_optional(self) -> None:
        req = IntakeAPIRequest(question="q")
        assert req.as_of_dt() is None

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_as_of_is_none(self, blank: str) -> None:
        req = IntakeAPIRequest(question="q", as_of=blank)
        assert req.as_of_dt() is None

    def test_as_of_parses_when_provided(self) -> None:
        req = IntakeAPIRequest(question="q", as_of="2024-06-01T00:00:00Z")
        parsed = req.as_of_dt()
        assert parsed is not None
        assert parsed.year == 2024

    def test_question_from_messages(self) -> None:
        req = IntakeAPIRequest(messages=(Message(role="user", content="Will Y happen?"),))
        assert req.resolved_question() == "Will Y happen?"

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValueError, match="no question"):
            IntakeAPIRequest().resolved_question()


class TestForecastJobSubmitRequest:
    def test_inherits_forecast_request_fields(self) -> None:
        req = ForecastJobSubmitRequest(
            question="q", as_of="2024-06-01T00:00:00Z", tier="delphi_deep", idempotency_key="k"
        )
        assert req.resolved_question() == "q"
        assert req.as_of_dt().year == 2024
        assert req.tier == "delphi_deep"

    def test_key_is_optional(self) -> None:
        req = ForecastJobSubmitRequest(question="q", as_of="2024-06-01T00:00:00Z")
        assert req.idempotency_key is None

    def test_empty_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForecastJobSubmitRequest(question="q", as_of="2024-06-01T00:00:00Z", idempotency_key="")

    def test_overlong_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForecastJobSubmitRequest(
                question="q", as_of="2024-06-01T00:00:00Z", idempotency_key="k" * 201
            )

    def test_job_payload_excludes_key_and_revalidates(self) -> None:
        req = ForecastJobSubmitRequest(
            question="q", as_of="2024-06-01T00:00:00Z", idempotency_key="k"
        )
        payload = req.job_payload()
        assert "idempotency_key" not in payload
        assert payload["question"] == "q"
        # The stored payload must round-trip into the execution-time request.
        parsed = ForecastAPIRequest.model_validate(payload)
        assert parsed.resolved_question() == "q"
        assert parsed.as_of_dt() == req.as_of_dt()


class TestBuildJobResponse:
    _T0 = datetime(2024, 6, 1, tzinfo=UTC)

    def test_full_job_maps_all_fields(self) -> None:
        job = ForecastJob(
            job_id="fj-1",
            status=JobStatus.SUCCEEDED,
            request={"question": "q"},
            idempotency_key="k",
            created_at=self._T0,
            started_at=self._T0,
            finished_at=self._T0,
            result={"object": "forecast.completion"},
        )
        dumped = build_job_response(job).model_dump(mode="json")
        assert dumped["object"] == "forecast.job"
        assert dumped["id"] == "fj-1"
        assert dumped["status"] == "succeeded"
        assert dumped["created"] == int(self._T0.timestamp())
        assert dumped["request"] == {"question": "q"}
        assert dumped["result"] == {"object": "forecast.completion"}
        assert dumped["error"] is None

    def test_pending_job_has_no_result(self) -> None:
        job = ForecastJob(job_id="fj-2", status=JobStatus.QUEUED, request={}, created_at=self._T0)
        response = build_job_response(job)
        assert response.status == "queued"
        assert response.result is None
        assert response.started_at is None
        assert response.finished_at is None

    def test_missing_created_at_maps_to_zero(self) -> None:
        job = ForecastJob(job_id="fj-3", status=JobStatus.QUEUED, request={})
        assert build_job_response(job).created == 0

    def test_failed_job_carries_error(self) -> None:
        job = ForecastJob(
            job_id="fj-4",
            status=JobStatus.FAILED,
            request={},
            created_at=self._T0,
            error="forecast_failed: boom",
        )
        response = build_job_response(job)
        assert response.status == "failed"
        assert response.error == "forecast_failed: boom"


class TestBuildClassifyResponse:
    def test_shape(self) -> None:
        req = IntakeAPIRequest(question="q")
        usage = usage_for("classify", model_calls=1)
        response = build_classify_response(req, _CLASSIFICATION, usage)
        dumped = response.model_dump(mode="json")
        assert dumped["object"] == "question.classification"
        assert dumped["classification"]["question_type"] == "binary"
        assert dumped["classification"]["entities"] == ["X"]
        assert dumped["classification"]["horizon"] == "2025"
        assert dumped["usage"]["tier"] == "classify"


class TestBuildFormalizeResponse:
    def test_accepted(self) -> None:
        outcome = IntakeOutcome(True, _CLASSIFICATION, _RESOLVABLE, None, None)
        response = build_formalize_response(
            IntakeAPIRequest(question="q"), outcome, usage_for("formalize", model_calls=2)
        )
        dumped = response.model_dump(mode="json")
        assert dumped["object"] == "question.formalization"
        assert not dumped["refused"]
        assert dumped["refusal_reason"] == ""
        assert dumped["formalized"]["text"] == _RESOLVABLE.text
        assert dumped["formalized"]["domain"] == "tech"
        assert dumped["formalized"]["resolution_criteria"] == _RESOLVABLE.resolution_criteria
        assert dumped["formalized"]["resolution_sources"] == ["vendor blog"]
        assert dumped["classification"]["question_type"] == "binary"

    def test_refused_omits_formalized(self) -> None:
        refusal = RefusalDecision(True, RefusalReason.UNDERSPECIFIED, "no criteria")
        outcome = IntakeOutcome(False, _CLASSIFICATION, _RESOLVABLE, refusal, None)
        response = build_formalize_response(
            IntakeAPIRequest(question="q"), outcome, usage_for("formalize", model_calls=2)
        )
        assert response.refused
        assert response.refusal_reason == "underspecified"
        assert response.refusal_detail == "no criteria"
        assert response.formalized is None

    def test_refusal_without_reason_maps_to_empty(self) -> None:
        outcome = IntakeOutcome(False, _CLASSIFICATION, None, RefusalDecision(True), None)
        response = build_formalize_response(
            IntakeAPIRequest(question="q"), outcome, usage_for("formalize", model_calls=2)
        )
        assert response.refused
        assert response.refusal_reason == ""
        assert response.formalized is None
