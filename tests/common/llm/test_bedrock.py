"""Unit tests for the Bedrock structured-output transport (§8). boto3 mocked."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import common.llm.bedrock as bedrock_mod
from common.llm import (
    BedrockStructuredClient,
    LLMConfig,
    LLMError,
    MalformedLLMOutput,
    StructuredPrompt,
)


def _resp(text: str) -> dict[str, Any]:
    return {"output": {"message": {"content": [{"text": text}]}}}


class _ThrottleError(Exception):
    def __init__(self) -> None:
        super().__init__("throttled")
        self.response = {"Error": {"Code": "ThrottlingException"}}


class ScriptedClient:
    """Fake converse client returning a scripted behavior per call."""

    def __init__(self, behaviors: Sequence[Any]) -> None:
        self._behaviors = list(behaviors)
        self.calls: list[dict[str, Any]] = []
        self._i = 0
        self._lock = threading.Lock()

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        with self._lock:
            self.calls.append(kwargs)
            beh = self._behaviors[min(self._i, len(self._behaviors) - 1)]
            self._i += 1
        if isinstance(beh, Exception):
            raise beh
        return _resp(beh)


class EchoClient:
    """Returns JSON echoing the user message so order can be asserted."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        user = kwargs["messages"][0]["content"][0]["text"]
        return _resp(json.dumps({"echo": user}))


def _client(behaviors: Sequence[Any], **cfg: Any) -> tuple[BedrockStructuredClient, ScriptedClient]:
    fake = ScriptedClient(behaviors)
    config = LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0, **cfg)
    return BedrockStructuredClient(model_id="model-x", client=fake, config=config), fake


class TestInvokeStructured:
    def test_happy_path_parses_json(self) -> None:
        client, fake = _client(['{"probability": 0.4}'])
        assert client.invoke_structured(system="s", user="u") == {"probability": 0.4}
        assert len(fake.calls) == 1

    def test_request_shaping(self) -> None:
        client, fake = _client(['{"x": 1}'], temperature=0.3, max_tokens=256, top_p=0.5)
        client.invoke_structured(system="sys", user="usr")
        call = fake.calls[0]
        assert call["modelId"] == "model-x"
        assert call["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.3, "topP": 0.5}
        assert call["system"] == [{"text": "sys"}]
        assert call["messages"][0]["content"][0]["text"] == "usr"

    def test_empty_system_omits_system_blocks(self) -> None:
        client, fake = _client(['{"x": 1}'])
        client.invoke_structured(system="", user="u")
        assert fake.calls[0]["system"] == []

    def test_strips_surrounding_prose(self) -> None:
        client, _ = _client(['Sure! Here is the answer: {"probability": 0.7}. Done.'])
        assert client.invoke_structured(system="s", user="u") == {"probability": 0.7}

    def test_malformed_then_success_retries(self) -> None:
        client, fake = _client(["not json at all", '{"ok": 1}'], max_retries=3)
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}
        assert len(fake.calls) == 2

    def test_malformed_exhausts_and_raises(self) -> None:
        client, fake = _client(["nope", "still nope", "nope3"], max_retries=2)
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured(system="s", user="u")
        assert len(fake.calls) == 2

    def test_non_object_json_is_malformed(self) -> None:
        client, _ = _client(["[1, 2, 3]"], max_retries=1)
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured(system="s", user="u")

    def test_throttling_is_retried(self) -> None:
        client, fake = _client([_ThrottleError(), '{"ok": 1}'], max_retries=3)
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}
        assert len(fake.calls) == 2

    def test_non_throttling_error_not_retried(self) -> None:
        client, fake = _client([ValueError("boom")], max_retries=3)
        with pytest.raises(LLMError):
            client.invoke_structured(system="s", user="u")
        assert len(fake.calls) == 1

    def test_missing_text_block_is_malformed(self) -> None:
        fake = ScriptedClient([])
        # Override to return a response with no text content block.
        fake.converse = lambda **kwargs: {"output": {"message": {"content": [{}]}}}  # type: ignore[method-assign]
        client = BedrockStructuredClient(model_id="m", client=fake, config=LLMConfig(max_retries=1))
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured(system="s", user="u")

    def test_missing_boto3_raises_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_name: str) -> Any:
            raise ModuleNotFoundError("boto3")

        monkeypatch.setattr(bedrock_mod.importlib, "import_module", _raise)
        client = BedrockStructuredClient(model_id="m")  # no injected client
        with pytest.raises(RuntimeError, match="boto3 is required"):
            client.invoke_structured(system="s", user="u")


class TestInvokeStructuredBatch:
    def test_empty_returns_empty(self) -> None:
        client, fake = _client(['{"x": 1}'])
        assert client.invoke_structured_batch([]) == []
        assert len(fake.calls) == 0

    def test_preserves_input_order(self) -> None:
        client = BedrockStructuredClient(model_id="m", client=EchoClient(), config=LLMConfig())
        prompts = [StructuredPrompt(system="s", user=f"u{i}", run_index=i) for i in range(6)]
        results = client.invoke_structured_batch(prompts)
        assert [r["echo"] for r in results] == [f"u{i}" for i in range(6)]

    def test_concurrency_is_bounded(self) -> None:
        state = {"active": 0, "max": 0}
        lock = threading.Lock()

        class TrackingClient:
            def converse(self, **kwargs: Any) -> Mapping[str, Any]:
                with lock:
                    state["active"] += 1
                    state["max"] = max(state["max"], state["active"])
                time.sleep(0.02)
                with lock:
                    state["active"] -= 1
                return _resp('{"ok": 1}')

        client = BedrockStructuredClient(
            model_id="m", client=TrackingClient(), config=LLMConfig(max_concurrency=2)
        )
        prompts = [StructuredPrompt(system="s", user=f"u{i}", run_index=i) for i in range(8)]
        client.invoke_structured_batch(prompts)
        assert state["max"] <= 2
        assert state["max"] >= 2  # genuine parallelism occurred

    def test_failure_propagates_from_batch(self) -> None:
        client, _ = _client(["bad", "bad", "bad"], max_retries=2)
        prompts = [StructuredPrompt(system="s", user="u", run_index=0)]
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured_batch(prompts)
