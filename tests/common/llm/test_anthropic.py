"""Unit tests for the direct Anthropic (Claude) API transport (§8).

The anthropic SDK + network + API key are all mocked/injected; nothing here
touches api.anthropic.com.
"""

from __future__ import annotations

import threading
import time
import types
from typing import Any

import pytest

import common.llm.anthropic_api as anthropic_mod
from common.llm import (
    AnthropicStructuredClient,
    LLMConfig,
    LLMError,
    MalformedLLMOutput,
    StructuredPrompt,
)
from common.secrets import EnvSecretProvider, SecretNotFoundError


def _resp(text: str) -> dict[str, Any]:
    """A Messages-API-shaped response with a single text block."""
    return {"content": [{"type": "text", "text": text}]}


class _RateLimitError(Exception):
    """Mimics anthropic.RateLimitError (carries an HTTP status_code)."""

    status_code = 429


class _OverloadedError(Exception):
    """Mimics anthropic.APIStatusError for the 529 'overloaded' condition."""


class ScriptedMessages:
    """Fake ``messages`` namespace returning a scripted behavior per ``create``."""

    def __init__(self, behaviors: list[Any]) -> None:
        self._behaviors = list(behaviors)
        self.calls: list[dict[str, Any]] = []
        self._i = 0
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        with self._lock:
            self.calls.append(kwargs)
            beh = self._behaviors[min(self._i, len(self._behaviors) - 1)]
            self._i += 1
        if isinstance(beh, Exception):
            raise beh
        return _resp(beh)


class EchoMessages:
    """Returns JSON echoing the user content so batch order can be asserted."""

    def create(self, **kwargs: Any) -> Any:
        import json

        return _resp(json.dumps({"echo": kwargs["messages"][0]["content"]}))


class FixedResponse:
    """A messages client that always returns the same raw response object."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def create(self, **_kwargs: Any) -> Any:
        return self._response


class _SdkBlock:
    """Mimics an anthropic SDK TextBlock (attribute access, not a mapping)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _SdkResponse:
    """Mimics an anthropic SDK Message (``.content`` list of blocks)."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content


def _client(behaviors: list[Any], **cfg: Any) -> tuple[AnthropicStructuredClient, ScriptedMessages]:
    fake = ScriptedMessages(behaviors)
    config = LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0, **cfg)
    return AnthropicStructuredClient(model_id="model-x", client=fake, config=config), fake


class TestInvokeStructured:
    def test_happy_path_parses_json(self) -> None:
        client, fake = _client(['{"probability": 0.4}'])
        assert client.invoke_structured(system="s", user="u") == {"probability": 0.4}
        assert len(fake.calls) == 1
        assert client.provider == "anthropic"
        assert client.model_id == "model-x"
        assert client.config.max_concurrency == 8

    def test_extracts_from_sdk_style_objects(self) -> None:
        # The real SDK returns objects: response.content[i].text (not mappings).
        fake = FixedResponse(_SdkResponse([_SdkBlock('{"ok": 1}')]))
        client = AnthropicStructuredClient(
            model_id="m", client=fake, config=LLMConfig(max_retries=1)
        )
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}

    def test_invalid_json_with_braces_is_malformed(self) -> None:
        client, _ = _client(['{"probability": }'], max_retries=1)
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured(system="s", user="u")

    def test_request_shaping(self) -> None:
        client, fake = _client(['{"x": 1}'], temperature=0.3, max_tokens=256, top_p=0.5)
        client.invoke_structured(system="sys", user="usr")
        call = fake.calls[0]
        assert call["model"] == "model-x"
        assert call["max_tokens"] == 256
        assert call["temperature"] == 0.3
        # top_p is intentionally NOT sent (adaptive-thinking models reject it).
        assert "top_p" not in call
        assert call["system"] == "sys"
        assert call["messages"] == [{"role": "user", "content": "usr"}]

    def test_no_thinking_or_effort_by_default(self) -> None:
        # Regression guard: with both knobs unset, the request keeps
        # temperature and carries neither thinking nor output_config.
        client, fake = _client(['{"x": 1}'])
        client.invoke_structured(system="s", user="u")
        call = fake.calls[0]
        assert "temperature" in call
        assert "thinking" not in call
        assert "output_config" not in call

    def test_adaptive_thinking_sets_param_and_omits_temperature(self) -> None:
        client, fake = _client(['{"x": 1}'], thinking="adaptive")
        client.invoke_structured(system="s", user="u")
        call = fake.calls[0]
        assert call["thinking"] == {"type": "adaptive"}
        # Adaptive-thinking models reject sampling params (same as top_p).
        assert "temperature" not in call
        assert "top_p" not in call

    def test_effort_sets_output_config(self) -> None:
        client, fake = _client(['{"x": 1}'], effort="high")
        client.invoke_structured(system="s", user="u")
        call = fake.calls[0]
        assert call["output_config"] == {"effort": "high"}
        # effort alone does not disable sampling or enable thinking
        assert "temperature" in call
        assert "thinking" not in call

    def test_thinking_and_effort_combine(self) -> None:
        client, fake = _client(['{"x": 1}'], thinking="adaptive", effort="max")
        client.invoke_structured(system="s", user="u")
        call = fake.calls[0]
        assert call["thinking"] == {"type": "adaptive"}
        assert call["output_config"] == {"effort": "max"}
        assert "temperature" not in call

    def test_thinking_block_before_text_still_parses(self) -> None:
        # A thinking-block dict followed by a text block: _extract_text must
        # skip the thinking block and return the JSON-bearing text.
        fake = FixedResponse(
            {
                "content": [
                    {"type": "thinking", "thinking": "let me reason about this..."},
                    {"type": "text", "text": '{"probability": 0.55}'},
                ]
            }
        )
        client = AnthropicStructuredClient(
            model_id="m",
            client=fake,
            config=LLMConfig(max_retries=1, thinking="adaptive", effort="high"),
        )
        assert client.invoke_structured(system="s", user="u") == {"probability": 0.55}

    def test_empty_system_omits_system_field(self) -> None:
        client, fake = _client(['{"x": 1}'])
        client.invoke_structured(system="", user="u")
        assert "system" not in fake.calls[0]

    def test_strips_surrounding_prose(self) -> None:
        client, _ = _client(['Sure! Here it is: {"probability": 0.7}. Done.'])
        assert client.invoke_structured(system="s", user="u") == {"probability": 0.7}

    def test_skips_non_text_blocks(self) -> None:
        # Adaptive-thinking models can emit a thinking block before the text.
        fake = FixedResponse(
            {
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": '{"ok": 1}'},
                ]
            }
        )
        client = AnthropicStructuredClient(
            model_id="m", client=fake, config=LLMConfig(max_retries=1)
        )
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}

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

    def test_missing_content_is_malformed(self) -> None:
        fake = FixedResponse({"content": []})
        client = AnthropicStructuredClient(
            model_id="m", client=fake, config=LLMConfig(max_retries=1)
        )
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured(system="s", user="u")

    def test_no_text_block_is_malformed(self) -> None:
        fake = FixedResponse({"content": [{"type": "thinking", "thinking": "x"}]})
        client = AnthropicStructuredClient(
            model_id="m", client=fake, config=LLMConfig(max_retries=1)
        )
        with pytest.raises(MalformedLLMOutput):
            client.invoke_structured(system="s", user="u")

    def test_status_code_throttle_is_retried(self) -> None:
        client, fake = _client([_RateLimitError(), '{"ok": 1}'], max_retries=3)
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}
        assert len(fake.calls) == 2

    def test_named_overloaded_error_is_retried(self) -> None:
        client, fake = _client([_OverloadedError(), '{"ok": 1}'], max_retries=3)
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}
        assert len(fake.calls) == 2

    def test_non_throttling_error_not_retried(self) -> None:
        client, fake = _client([ValueError("boom")], max_retries=3)
        with pytest.raises(LLMError):
            client.invoke_structured(system="s", user="u")
        assert len(fake.calls) == 1


class TestBatch:
    def test_empty_returns_empty(self) -> None:
        client, fake = _client(['{"x": 1}'])
        assert client.invoke_structured_batch([]) == []
        assert len(fake.calls) == 0

    def test_preserves_input_order(self) -> None:
        client = AnthropicStructuredClient(model_id="m", client=EchoMessages(), config=LLMConfig())
        prompts = [StructuredPrompt(system="s", user=f"u{i}", run_index=i) for i in range(6)]
        results = client.invoke_structured_batch(prompts)
        assert [r["echo"] for r in results] == [f"u{i}" for i in range(6)]

    def test_concurrency_is_bounded(self) -> None:
        state = {"active": 0, "max": 0}
        lock = threading.Lock()

        class TrackingMessages:
            def create(self, **_kwargs: Any) -> Any:
                with lock:
                    state["active"] += 1
                    state["max"] = max(state["max"], state["active"])
                time.sleep(0.02)
                with lock:
                    state["active"] -= 1
                return _resp('{"ok": 1}')

        client = AnthropicStructuredClient(
            model_id="m", client=TrackingMessages(), config=LLMConfig(max_concurrency=2)
        )
        prompts = [StructuredPrompt(system="s", user=f"u{i}", run_index=i) for i in range(8)]
        client.invoke_structured_batch(prompts)
        assert state["max"] == 2  # bounded, and genuine parallelism occurred


class TestApiKeyResolution:
    @staticmethod
    def _fake_module(captured: dict[str, Any]) -> Any:
        messages = ScriptedMessages(['{"ok": 1}'])

        class _Client:
            def __init__(self, *, api_key: str) -> None:
                captured["api_key"] = api_key
                self.messages = messages

        captured["messages"] = messages
        return types.SimpleNamespace(Anthropic=_Client)

    def test_resolves_key_from_injected_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        module = self._fake_module(captured)
        monkeypatch.setattr(anthropic_mod.importlib, "import_module", lambda _name: module)
        client = AnthropicStructuredClient(
            model_id="m",
            secrets=EnvSecretProvider({"DELPHI_SECRET_ANTHROPIC_API_KEY": "sk-xyz"}),
            config=LLMConfig(max_retries=1),
        )
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}
        assert captured["api_key"] == "sk-xyz"
        assert captured["messages"].calls[0]["model"] == "m"

    def test_resolves_key_from_default_env_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        module = self._fake_module(captured)
        monkeypatch.setattr(anthropic_mod.importlib, "import_module", lambda _name: module)
        monkeypatch.setenv("DELPHI_SECRET_ANTHROPIC_API_KEY", "sk-env")
        client = AnthropicStructuredClient(model_id="m", config=LLMConfig(max_retries=1))
        assert client.invoke_structured(system="s", user="u") == {"ok": 1}
        assert captured["api_key"] == "sk-env"

    def test_missing_key_raises(self) -> None:
        client = AnthropicStructuredClient(
            model_id="m", secrets=EnvSecretProvider({}), config=LLMConfig(max_retries=1)
        )
        with pytest.raises(SecretNotFoundError):
            client.invoke_structured(system="s", user="u")

    def test_missing_sdk_raises_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_name: str) -> Any:
            raise ModuleNotFoundError("anthropic")

        monkeypatch.setattr(anthropic_mod.importlib, "import_module", _raise)
        client = AnthropicStructuredClient(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="anthropic is required"):
            client.invoke_structured(system="s", user="u")


class CountingRefusalMessages:
    """Always returns a safety-refusal response and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: Any) -> Any:
        self.calls += 1
        return {
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {"category": "bio"},
        }


class TestRefusal:
    def test_refusal_raises_typed_error_and_is_not_retried(self) -> None:
        from common.llm import LLMRefusedError

        fake = CountingRefusalMessages()
        config = LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0)
        client = AnthropicStructuredClient(model_id="model-x", client=fake, config=config)
        with pytest.raises(LLMRefusedError, match="category='bio'"):
            client.invoke_structured(system="s", user="u")
        assert fake.calls == 1  # a refusal must never be retried

    def test_sdk_shaped_refusal_detected(self) -> None:
        from common.llm import LLMRefusedError

        details = types.SimpleNamespace(category="cyber")
        response = types.SimpleNamespace(content=[], stop_reason="refusal", stop_details=details)
        client = AnthropicStructuredClient(
            model_id="model-x",
            client=FixedResponse(response),
            config=LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0),
        )
        with pytest.raises(LLMRefusedError, match="cyber"):
            client.invoke_structured(system="s", user="u")

    def test_refusal_without_details_still_typed(self) -> None:
        from common.llm import LLMRefusedError

        client = AnthropicStructuredClient(
            model_id="model-x",
            client=FixedResponse({"content": [], "stop_reason": "refusal"}),
            config=LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0),
        )
        with pytest.raises(LLMRefusedError, match="category=None"):
            client.invoke_structured(system="s", user="u")
