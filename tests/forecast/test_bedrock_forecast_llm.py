"""Unit tests for BedrockForecastLLM (§8). boto3 mocked via injected client."""

from __future__ import annotations

from typing import Any

import pytest

from common.llm import BedrockStructuredClient, LLMConfig, MalformedLLMOutput
from core.forecast.llm import BedrockForecastLLM, ForecastRequest


class FixedClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.users: list[str] = []
        self.systems: list[str] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.users.append(kwargs["messages"][0]["content"][0]["text"])
        self.systems.append(kwargs["system"][0]["text"] if kwargs["system"] else "")
        return {"output": {"message": {"content": [{"text": self.text}]}}}


def _llm(text: str) -> tuple[BedrockForecastLLM, FixedClient]:
    fake = FixedClient(text)
    client = BedrockStructuredClient(
        model_id="model-x", client=fake, config=LLMConfig(max_retries=1)
    )
    return BedrockForecastLLM(client), fake


def _reqs(n: int) -> list[ForecastRequest]:
    return [
        ForecastRequest(content="doc body", content_hash="h1", run_index=i, prompt="Will it work?")
        for i in range(n)
    ]


def test_happy_path_returns_n_draws() -> None:
    llm, _ = _llm('{"probability": 0.6}')
    draws = llm.forecast_batch(_reqs(3))
    assert len(draws) == 3
    assert all(d.probability == pytest.approx(0.6) for d in draws)
    assert [d.run_index for d in draws] == [0, 1, 2]
    assert llm.model_version == "model-x"
    assert llm.prompt_version == "delphi_forecast_v1"
    assert draws[0].provenance["provider"] == "bedrock"


def test_empty_requests_returns_empty() -> None:
    llm, fake = _llm('{"probability": 0.5}')
    assert llm.forecast_batch([]) == ()
    assert fake.users == []


def test_prompt_and_content_reach_the_model() -> None:
    llm, fake = _llm('{"probability": 0.5}')
    llm.forecast_batch(_reqs(1))
    assert "Will it work?" in fake.users[0]
    assert "doc body" in fake.users[0]
    assert "JSON" in fake.systems[0]


def test_probability_out_of_range_is_malformed() -> None:
    llm, _ = _llm('{"probability": 1.5}')
    with pytest.raises(MalformedLLMOutput):
        llm.forecast_batch(_reqs(1))


def test_missing_probability_key_is_malformed() -> None:
    llm, _ = _llm('{"foo": 0.5}')
    with pytest.raises(MalformedLLMOutput):
        llm.forecast_batch(_reqs(1))


def test_nan_probability_is_malformed() -> None:
    llm, _ = _llm('{"probability": "NaN"}')
    with pytest.raises(MalformedLLMOutput):
        llm.forecast_batch(_reqs(1))
