"""Derived semantic recall index over the experiment registry."""

from core.memory.embedder import BedrockEmbedder, DeterministicEmbedder, Embedder
from core.memory.index import (
    IndexDocument,
    InMemoryVectorIndex,
    MemoryError,
    PostgresVectorIndex,
    RecallOutcome,
    Recollection,
    VectorIndex,
    assemble_document,
    index_experiment,
    render_spec_description,
)
from core.memory.recall import MemoryRecall

__all__ = [
    "BedrockEmbedder",
    "DeterministicEmbedder",
    "Embedder",
    "InMemoryVectorIndex",
    "IndexDocument",
    "MemoryError",
    "MemoryRecall",
    "PostgresVectorIndex",
    "RecallOutcome",
    "Recollection",
    "VectorIndex",
    "assemble_document",
    "index_experiment",
    "render_spec_description",
]
