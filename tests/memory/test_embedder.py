"""Unit tests for the memory embedders (§8). The Bedrock client is mocked."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from common.llm.embedding import BedrockEmbeddingClient
from common.llm.errors import MalformedLLMOutput
from core.memory.embedder import BedrockEmbedder, DeterministicEmbedder, Embedder
from core.memory.index import DimensionMismatchError, InMemoryVectorIndex


class HashInvokeClient:
    """Deterministic fake: fixed-length vector derived from inputText."""

    def __init__(self, *, dimensions: int) -> None:
        self._dim = dimensions

    def invoke_model(self, **kwargs: Any) -> Mapping[str, Any]:
        text = json.loads(kwargs["body"])["inputText"]
        seed = float(len(text))
        vector = [seed + i for i in range(self._dim)]
        return {"body": json.dumps({"embedding": vector}).encode("utf-8")}


def _bedrock_embedder(dimensions: int = 256) -> BedrockEmbedder:
    client = BedrockEmbeddingClient(
        dimensions=dimensions, client=HashInvokeClient(dimensions=dimensions)
    )
    return BedrockEmbedder(client)


def test_satisfies_embedder_protocol() -> None:
    assert isinstance(_bedrock_embedder(), Embedder)


def test_dim_reflects_client_dimension() -> None:
    assert _bedrock_embedder(dimensions=512).dim == 512


def test_embed_returns_vector_per_text() -> None:
    embedder = _bedrock_embedder(dimensions=256)
    vectors = embedder.embed(["alpha", "beta gamma"])
    assert len(vectors) == 2
    assert all(len(v) == 256 for v in vectors)


def test_embed_is_deterministic() -> None:
    embedder = _bedrock_embedder()
    assert embedder.embed(["stable text"]) == embedder.embed(["stable text"])


def test_embed_empty_returns_empty() -> None:
    assert _bedrock_embedder().embed([]) == []


def test_dimension_mismatch_surfaced_by_index() -> None:
    """VectorIndex._embed_one guards against a client/index dim disagreement."""

    class LyingClient:
        def invoke_model(self, **kwargs: Any) -> Mapping[str, Any]:
            return {"body": json.dumps({"embedding": [1.0, 2.0]}).encode("utf-8")}

    # Claims dim 256 but the transport would only accept a length-256 vector, so
    # the malformed short vector is rejected before it ever reaches the index.
    client = BedrockEmbeddingClient(dimensions=256, client=LyingClient())
    embedder = BedrockEmbedder(client)
    index = InMemoryVectorIndex(store=_NullRegistry(), embedder=embedder)  # type: ignore[arg-type]
    with pytest.raises(MalformedLLMOutput):  # short vector rejected before the index
        index._embed_one("text")  # noqa: SLF001 - exercising the guard directly


def test_deterministic_embedder_dim_mismatch_raises() -> None:
    """A too-short deterministic vector is caught by the index dim guard."""

    class ShortEmbedder:
        @property
        def dim(self) -> int:
            return 128

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0]] * len(texts)

    index = InMemoryVectorIndex(store=_NullRegistry(), embedder=ShortEmbedder())  # type: ignore[arg-type]
    with pytest.raises(DimensionMismatchError):
        index._embed_one("text")  # noqa: SLF001


def test_deterministic_and_bedrock_are_interchangeable() -> None:
    assert isinstance(DeterministicEmbedder(dim=256), Embedder)
    assert isinstance(_bedrock_embedder(256), Embedder)


class _NullRegistry:
    """Minimal registry stand-in; _embed_one never touches it."""
