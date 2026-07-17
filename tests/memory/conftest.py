"""Shared fixtures for agent memory tests.

Time and embeddings are deterministic; no wall-clock or network is used.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from core.memory.embedder import DeterministicEmbedder, Embedder
from core.memory.index import InMemoryVectorIndex, PostgresVectorIndex
from core.memory.recall import MemoryRecall
from core.registry.models import DecisionInput, ResultInput
from core.registry.store import InMemoryRegistryStore, PostgresRegistryStore
from tests.conftest import postgres_test_dsn
from tests.registry.conftest import IncrementingClock, make_experiment_input, make_repro

CLOCK_START = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


class ClusteredFixtureEmbedder:
    """Deterministic embedder that clusters elections vs weather vocabulary."""

    def __init__(self, *, dim: int = 8) -> None:
        self._dim = dim
        self._elections = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:dim])
        self._weather = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:dim])
        self._neutral = _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0][:dim])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "election" in lowered or "poll" in lowered or "incumbent" in lowered:
                vectors.append(self._elections.copy())
            elif "weather" in lowered or "hurricane" in lowered:
                vectors.append(self._weather.copy())
            else:
                vectors.append(self._neutral.copy())
        return vectors


def _unit(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


@pytest.fixture
def clock() -> IncrementingClock:
    return IncrementingClock(start=CLOCK_START)


@pytest.fixture
def store(clock: IncrementingClock) -> InMemoryRegistryStore:
    return InMemoryRegistryStore(clock=clock)


@pytest.fixture
def embedder() -> Embedder:
    return ClusteredFixtureEmbedder()


@pytest.fixture
def memory_index(store: InMemoryRegistryStore, embedder: Embedder) -> InMemoryVectorIndex:
    return InMemoryVectorIndex(store, embedder)


@pytest.fixture
def recall(
    embedder: Embedder,
    memory_index: InMemoryVectorIndex,
    store: InMemoryRegistryStore,
) -> MemoryRecall:
    return MemoryRecall(embedder, memory_index, store)


def record_experiment_with_outcome(
    store: InMemoryRegistryStore,
    *,
    hypothesis: str,
    niche: str,
    outcome: str,
    rationale: str,
    spec_hash: str = "spec-hash-001",
    params: dict[str, Any] | None = None,
) -> str:
    """Record experiment + result + decision; return experiment id."""
    exp = make_experiment_input(
        hypothesis=hypothesis,
        niche=niche,
        repro=make_repro(spec_hash=spec_hash, params=params or {"lookback": 20}),
    )
    exp_id = store.record_experiment(exp)
    store.record_result(
        ResultInput(
            experiment_id=exp_id,
            status="success",
            metrics={"sharpe": 0.5},
        )
    )
    store.record_decision(
        DecisionInput(
            experiment_id=exp_id,
            outcome=outcome,  # type: ignore[arg-type]
            deciding_component="harness.gates",
            component_version="1.0.0",
            rationale=rationale,
            evidence={"gate": "deflated_sharpe"},
        )
    )
    return exp_id


@pytest.fixture
def postgres_memory_stack() -> Iterator[
    tuple[PostgresRegistryStore, PostgresVectorIndex, MemoryRecall]
]:
    dsn = postgres_test_dsn()
    clock = IncrementingClock(start=CLOCK_START)
    registry = PostgresRegistryStore.connect(dsn, migrate=True, clock=clock)
    embedder = DeterministicEmbedder(dim=128)
    index = PostgresVectorIndex.connect(dsn, registry, embedder, migrate=True, clock=clock)
    recall = MemoryRecall(embedder, index, registry)
    try:
        with registry._conn.cursor() as cur:  # noqa: SLF001 — test cleanup only
            cur.execute("TRUNCATE registry_events, memory_index RESTART IDENTITY")
        registry._conn.commit()
        yield registry, index, recall
    finally:
        index.close()
        registry.close()
