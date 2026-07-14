"""Semantic recall API over the derived memory index.

Contract: read-only over the registry. Near-duplicate detection is ADVISORY
only — honest trial counting remains exact-fingerprint (registry/06).
"""

from __future__ import annotations

from core.memory.embedder import Embedder
from core.memory.index import RecallOutcome, Recollection, VectorIndex
from core.registry.store import RegistryStore


class MemoryRecall:
    """Assemble relevant prior experiments for agent context injection."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        store: RegistryStore,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._store = store

    def recall(
        self,
        *,
        query: str,
        niche: str | None = None,
        outcome: RecallOutcome = "any",
        k: int = 10,
    ) -> list[Recollection]:
        """Return up to ``k`` registry experiments relevant to ``query``.

        Contract: optionally filtered by niche/outcome (failures included).
        Deterministic given a fixed embedder. Read-only over the registry; never
        mutates it.
        """
        if not query.strip():
            msg = "Recall query must be non-empty."
            raise ValueError(msg)
        if k < 0:
            msg = "k must be non-negative."
            raise ValueError(msg)
        vector = self._embedder.embed([query])[0]
        return self._index.search(vector, niche=niche, outcome=outcome, k=k)

    def lessons(
        self,
        *,
        query: str,
        niche: str | None = None,
        k: int = 10,
    ) -> list[str]:
        """Return distilled lessons from the most relevant prior experiments."""
        recollections = self.recall(query=query, niche=niche, outcome="any", k=k)
        return [item.lessons for item in recollections if item.lessons]

    def near_duplicates(
        self,
        *,
        spec_description: str,
        threshold: float,
    ) -> list[Recollection]:
        """ADVISORY: prior experiments semantically close to a candidate spec.

        Helps the agent avoid redundant work. NOT a trial-accounting mechanism —
        counting is by exact ``trial_fingerprint`` (registry prompts 03/06).
        """
        if not spec_description.strip():
            msg = "spec_description must be non-empty."
            raise ValueError(msg)
        if not 0.0 <= threshold <= 1.0:
            msg = "threshold must be between 0.0 and 1.0."
            raise ValueError(msg)
        vector = self._embedder.embed([spec_description])[0]
        candidates = self._index.search(vector, niche=None, outcome="any", k=100)
        return [item for item in candidates if item.score >= threshold]
