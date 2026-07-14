"""Unit tests for the Bedrock Titan embedding transport (§8). boto3 mocked."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import common.llm.embedding as embedding_mod
from common.llm import LLMConfig, LLMError, MalformedLLMOutput
from common.llm.embedding import AMAZON_TITAN_EMBED_V2_ID, BedrockEmbeddingClient


class _StreamingBody:
    """Mimics the boto3 StreamingBody: a one-shot ``read()`` of JSON bytes."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data


class _ThrottleError(Exception):
    def __init__(self) -> None:
        super().__init__("throttled")
        self.response = {"Error": {"Code": "ThrottlingException"}}


def _embedding_response(vector: list[float]) -> dict[str, Any]:
    return {"body": _StreamingBody({"embedding": vector, "inputTextTokenCount": 3})}


class ScriptedInvokeClient:
    """Fake invoke_model client returning a scripted behavior per call."""

    def __init__(self, behaviors: Sequence[Any]) -> None:
        self._behaviors = list(behaviors)
        self.calls: list[dict[str, Any]] = []
        self._i = 0
        self._lock = threading.Lock()

    def invoke_model(self, **kwargs: Any) -> Mapping[str, Any]:
        with self._lock:
            self.calls.append(kwargs)
            beh = self._behaviors[min(self._i, len(self._behaviors) - 1)]
            self._i += 1
        if isinstance(beh, Exception):
            raise beh
        return beh


class HashInvokeClient:
    """Deterministic: returns a fixed-length vector derived from inputText."""

    def __init__(self, *, dimensions: int) -> None:
        self._dim = dimensions

    def invoke_model(self, **kwargs: Any) -> Mapping[str, Any]:
        text = json.loads(kwargs["body"])["inputText"]
        seed = float(sum(ord(c) for c in text))
        vector = [seed + i for i in range(self._dim)]
        return _embedding_response(vector)


def _client(
    behaviors: Sequence[Any], *, dimensions: int = 256, **cfg: Any
) -> tuple[BedrockEmbeddingClient, ScriptedInvokeClient]:
    fake = ScriptedInvokeClient(behaviors)
    config = LLMConfig(retry_backoff_base=0.0, retry_backoff_max=0.0, **cfg)
    client = BedrockEmbeddingClient(dimensions=dimensions, client=fake, config=config)
    return client, fake


class TestConstruction:
    def test_default_model_id_and_dim(self) -> None:
        client = BedrockEmbeddingClient(client=ScriptedInvokeClient([]))
        assert client.model_id == AMAZON_TITAN_EMBED_V2_ID
        assert client.dimensions == 1024

    def test_unsupported_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be one of"):
            BedrockEmbeddingClient(dimensions=128)


class TestEmbed:
    def test_happy_path_returns_vector(self) -> None:
        vector = [float(i) for i in range(256)]
        client, fake = _client([_embedding_response(vector)], dimensions=256)
        assert client.embed(["hello"]) == [vector]
        assert len(fake.calls) == 1

    def test_request_shaping(self) -> None:
        vector = [float(i) for i in range(256)]
        client, fake = _client([_embedding_response(vector)], dimensions=256)
        client.embed(["what is inflation"])
        body = json.loads(fake.calls[0]["body"])
        assert fake.calls[0]["modelId"] == AMAZON_TITAN_EMBED_V2_ID
        assert body == {"inputText": "what is inflation", "dimensions": 256, "normalize": True}

    def test_empty_returns_empty(self) -> None:
        client, fake = _client([_embedding_response([1.0])])
        assert client.embed([]) == []
        assert len(fake.calls) == 0

    def test_preserves_input_order(self) -> None:
        client = BedrockEmbeddingClient(dimensions=256, client=HashInvokeClient(dimensions=256))
        texts = [f"text-{i}" for i in range(6)]
        results = client.embed(texts)
        expected = [[float(sum(ord(c) for c in t)) + i for i in range(256)] for t in texts]
        assert results == expected

    def test_deterministic_for_same_input(self) -> None:
        client = BedrockEmbeddingClient(dimensions=256, client=HashInvokeClient(dimensions=256))
        assert client.embed(["stable"]) == client.embed(["stable"])

    def test_concurrency_is_bounded(self) -> None:
        state = {"active": 0, "max": 0}
        lock = threading.Lock()

        class TrackingClient:
            def invoke_model(self, **kwargs: Any) -> Mapping[str, Any]:
                with lock:
                    state["active"] += 1
                    state["max"] = max(state["max"], state["active"])
                time.sleep(0.02)
                with lock:
                    state["active"] -= 1
                return _embedding_response([float(i) for i in range(256)])

        client = BedrockEmbeddingClient(
            dimensions=256, client=TrackingClient(), config=LLMConfig(max_concurrency=2)
        )
        client.embed([f"t{i}" for i in range(8)])
        assert state["max"] == 2


class TestErrors:
    def test_throttling_is_retried(self) -> None:
        vector = [float(i) for i in range(256)]
        client, fake = _client([_ThrottleError(), _embedding_response(vector)], max_retries=3)
        assert client.embed(["x"]) == [vector]
        assert len(fake.calls) == 2

    def test_non_throttling_error_not_retried(self) -> None:
        client, fake = _client([ValueError("boom")], max_retries=3)
        with pytest.raises(LLMError):
            client.embed(["x"])
        assert len(fake.calls) == 1

    def test_missing_embedding_is_malformed(self) -> None:
        client, _ = _client([{"body": _StreamingBody({"inputTextTokenCount": 1})}], max_retries=1)
        with pytest.raises(MalformedLLMOutput):
            client.embed(["x"])

    def test_wrong_dimension_is_malformed(self) -> None:
        client, _ = _client([_embedding_response([1.0, 2.0])], dimensions=256, max_retries=1)
        with pytest.raises(MalformedLLMOutput, match="dimension"):
            client.embed(["x"])

    def test_non_json_body_is_malformed(self) -> None:
        class BadBody:
            def read(self) -> bytes:
                return b"not json"

        client, _ = _client([{"body": BadBody()}], max_retries=1)
        with pytest.raises(MalformedLLMOutput):
            client.embed(["x"])

    def test_missing_body_is_malformed(self) -> None:
        client, _ = _client([{"no_body": 1}], max_retries=1)
        with pytest.raises(MalformedLLMOutput):
            client.embed(["x"])

    def test_missing_boto3_raises_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_name: str) -> Any:
            raise ModuleNotFoundError("boto3")

        monkeypatch.setattr(embedding_mod.importlib, "import_module", _raise)
        client = BedrockEmbeddingClient(dimensions=256)  # no injected client
        with pytest.raises(RuntimeError, match="boto3 is required"):
            client.embed(["x"])
