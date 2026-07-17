"""Shared fixtures for orchestration-primitive tests (Delphi seed subset)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from core.orchestration.run_state import InMemoryRunStateStore, PostgresRunStateStore
from tests.conftest import postgres_test_dsn
from tests.registry.conftest import IncrementingClock

CLOCK_START = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def memory_run_state() -> InMemoryRunStateStore:
    return InMemoryRunStateStore()


@pytest.fixture
def postgres_run_state() -> Iterator[PostgresRunStateStore]:
    dsn = postgres_test_dsn()
    store = PostgresRunStateStore.connect(dsn, migrate=True, clock=IncrementingClock())
    try:
        with store._conn.cursor() as cur:  # noqa: SLF001 — test cleanup
            cur.execute("TRUNCATE orchestration_runs RESTART IDENTITY")
        store._conn.commit()
        yield store
    finally:
        store.close()
