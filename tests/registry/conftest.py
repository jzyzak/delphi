"""Shared fixtures and builders for registry tests.

Time is injected and deterministic; no wall-clock or network is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from core.registry.models import (
    DataSnapshot,
    EnvFingerprint,
    ExperimentInput,
    ReproMetadata,
)
from core.registry.store import InMemoryRegistryStore, PostgresRegistryStore
from tests.conftest import postgres_test_dsn

AS_OF = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
CLOCK_START = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


class IncrementingClock:
    """Deterministic clock advancing one second per call (thread-unsafe by design)."""

    def __init__(self, start: datetime = CLOCK_START, step_seconds: int = 1) -> None:
        self._t = start
        self._step = timedelta(seconds=step_seconds)

    def __call__(self) -> datetime:
        current = self._t
        self._t += self._step
        return current


def make_repro(**overrides: Any) -> ReproMetadata:
    """Build a complete, valid reproducibility bundle for tests."""
    base: dict[str, Any] = {
        "code_sha": "a1b2c3d4",
        "dirty": False,
        "spec_kind": "dsl",
        "spec_hash": "spec-hash-001",
        "params": {"lookback": 20, "z_entry": 1.5},
        "data_snapshot": DataSnapshot(
            as_of=AS_OF,
            universe_spec={"classification": "elections", "min_price": 1.0},
        ),
        "env": EnvFingerprint(
            python_version="3.12.3",
            packages={"polars": "1.0.0", "numpy": "2.0.0"},
            image_digest="sha256:deadbeef",
        ),
        "seeds": {"numpy": 7, "python": 13},
    }
    base.update(overrides)
    return ReproMetadata(**base)


def make_experiment_input(**overrides: Any) -> ExperimentInput:
    """Build a valid experiment input bundle for tests."""
    base: dict[str, Any] = {
        "hypothesis": "Base rates beat inside-view estimates on rare events.",
        "economic_rationale": "Under-covered events resolve near their base rates.",
        "author": "agent.event_forecaster",
        "niche": "us_elections",
        "repro": make_repro(),
        "parent_experiment_id": None,
    }
    base.update(overrides)
    return ExperimentInput(**base)


@pytest.fixture
def store() -> InMemoryRegistryStore:
    return InMemoryRegistryStore(clock=IncrementingClock())


@pytest.fixture
def postgres_store() -> Iterator[PostgresRegistryStore]:
    dsn = postgres_test_dsn()
    store = PostgresRegistryStore.connect(dsn, migrate=True, clock=IncrementingClock())
    try:
        with store._conn.cursor() as cur:  # noqa: SLF001 — test cleanup only
            cur.execute("TRUNCATE registry_events RESTART IDENTITY")
        store._conn.commit()
        yield store
    finally:
        store.close()
