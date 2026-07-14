"""Shared fixtures for orchestration-primitive tests (Delphi seed subset)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from core.orchestration.run_state import InMemoryRunStateStore, PostgresRunStateStore
from tests.registry.conftest import IncrementingClock

CLOCK_START = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _postgres_dsn() -> str | None:
    return os.environ.get("DELPHI_PG_DSN")


@pytest.fixture
def memory_run_state() -> InMemoryRunStateStore:
    return InMemoryRunStateStore()


@pytest.fixture
def postgres_run_state() -> Iterator[PostgresRunStateStore]:
    dsn = _postgres_dsn()
    if not dsn:
        pytest.skip("DELPHI_PG_DSN not set")
    store = PostgresRunStateStore.connect(dsn, migrate=True, clock=IncrementingClock())
    try:
        with store._conn.cursor() as cur:  # noqa: SLF001 — test cleanup
            cur.execute("TRUNCATE orchestration_runs RESTART IDENTITY")
        store._conn.commit()
        yield store
    finally:
        store.close()
