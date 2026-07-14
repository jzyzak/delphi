"""Unit tests for BedrockEvidenceLikelihoodLLM (§8). boto3 mocked.

This seam must elicit an evidence log-likelihood-ratio and reject absolute
probabilities (CLAUDE.md sections 10/14).
"""

from __future__ import annotations

from typing import Any

import pytest

from common.llm import BedrockStructuredClient, LLMConfig, MalformedLLMOutput
from core.forecast.bayesian import BedrockEvidenceLikelihoodLLM, LikelihoodRequest


class FixedClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.users: list[str] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.users.append(kwargs["messages"][0]["content"][0]["text"])
        return {"output": {"message": {"content": [{"text": self.text}]}}}


def _llm(text: str) -> tuple[BedrockEvidenceLikelihoodLLM, FixedClient]:
    fake = FixedClient(text)
    client = BedrockStructuredClient(
        model_id="model-x", client=fake, config=LLMConfig(max_retries=1)
    )
    return BedrockEvidenceLikelihoodLLM(client), fake


def _reqs(n: int, *, prior: float = 0.3) -> list[LikelihoodRequest]:
    return [
        LikelihoodRequest(
            content="evidence body",
            content_hash="h1",
            run_index=i,
            prompt="Weigh the evidence.",
            prior=prior,
        )
        for i in range(n)
    ]


def test_happy_path_returns_log_lr_draws() -> None:
    llm, _ = _llm('{"evidence_log_lr": 0.5}')
    draws = llm.elicit_log_lr_batch(_reqs(3))
    assert len(draws) == 3
    assert all(d.evidence_log_lr == pytest.approx(0.5) for d in draws)
    assert llm.prompt_version == "evidence_log_lr_v1"
    assert draws[0].provenance["elicitation"] == "evidence_log_lr"
    assert draws[0].provenance["prior"] == pytest.approx(0.3)


def test_empty_requests_returns_empty() -> None:
    llm, fake = _llm('{"evidence_log_lr": 0.0}')
    assert llm.elicit_log_lr_batch([]) == ()
    assert fake.users == []


def test_prior_is_included_in_prompt() -> None:
    llm, fake = _llm('{"evidence_log_lr": 0.0}')
    llm.elicit_log_lr_batch(_reqs(1, prior=0.42))
    assert "Prior probability: 0.42" in fake.users[0]
    assert "evidence body" in fake.users[0]


def test_non_finite_log_lr_is_malformed() -> None:
    llm, _ = _llm('{"evidence_log_lr": Infinity}')
    with pytest.raises(MalformedLLMOutput):
        llm.elicit_log_lr_batch(_reqs(1))


def test_absolute_probability_payload_is_rejected() -> None:
    # A payload shaped like an absolute probability lacks evidence_log_lr.
    llm, _ = _llm('{"probability": 0.7}')
    with pytest.raises(MalformedLLMOutput):
        llm.elicit_log_lr_batch(_reqs(1))
