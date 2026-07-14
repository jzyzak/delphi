"""Embedding interface and deterministic local implementation.

Contract: embedders are pure, mockable, and deterministic. Tests never call a
live model; production may swap in a model-backed implementation behind the
same ``Embedder`` protocol.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from common.llm.embedding import BedrockEmbeddingClient

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Maps text batches to fixed-dimension unit vectors."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input text."""
        ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class DeterministicEmbedder:
    """Hash-based token n-gram embedder: reproducible, offline, no network.

    Contract: identical text always yields an identical unit vector. Suitable as
    the default concrete implementation and as a stand-in for model embeddings in
    tests.
    """

    def __init__(self, *, dim: int = 128) -> None:
        if dim < 8:
            msg = "Embedding dimension must be at least 8."
            raise ValueError(msg)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float64)
            tokens = _tokenize(text)
            for idx, token in enumerate(tokens):
                bucket = _hash_bucket(token, self._dim)
                vec[bucket] += 1.0
                if idx > 0:
                    bigram = f"{tokens[idx - 1]} {token}"
                    vec[_hash_bucket(bigram, self._dim)] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


def _hash_bucket(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big") % dim


class BedrockEmbedder:
    """Model-backed ``Embedder`` over Amazon Titan Text Embeddings V2.

    Contract: a drop-in for :class:`DeterministicEmbedder` behind the
    :class:`Embedder` protocol, delegating batches to a
    :class:`~common.llm.embedding.BedrockEmbeddingClient`. ``dim`` is the client's
    configured output dimension and MUST match the pgvector column (the memory
    migration is rendered from this value). The deterministic embedder remains
    the offline floor; this is the higher-quality production option.
    """

    def __init__(self, client: BedrockEmbeddingClient) -> None:
        self._client = client

    @property
    def dim(self) -> int:
        return self._client.dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._client.embed(texts)
