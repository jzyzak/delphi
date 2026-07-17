"""Shared fixtures for PIT store tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from core.pit.adapters.fixtures import ingest_synthetic_bars, utc_dt
from core.pit.store import InMemoryPitStore, PitStore, PostgresPitStore
from tests.conftest import TEST_PG_DSN_ENV_VAR, postgres_test_dsn

T0 = utc_dt(2024, 1, 1)
T1 = utc_dt(2024, 1, 5)
T2 = utc_dt(2024, 1, 10)
T3 = utc_dt(2024, 1, 15)
KNOWLEDGE_INITIAL = utc_dt(2024, 1, 6, 16, 0)


@pytest.fixture
def memory_store() -> InMemoryPitStore:
    return InMemoryPitStore()


def _postgres_available() -> bool:
    # Collection-time availability probe (no skip/fail semantics): reads the
    # dedicated TEST DSN only — fixtures here truncate tables.
    dsn = os.environ.get(TEST_PG_DSN_ENV_VAR)
    if not dsn:
        return False
    try:
        with PostgresPitStore.connect(dsn, migrate=True) as store:
            store.append([])
    except Exception:
        return False
    return True


@pytest.fixture
def postgres_store() -> Iterator[PostgresPitStore]:
    dsn = postgres_test_dsn()
    store = PostgresPitStore.connect(dsn, migrate=True)
    try:
        with store._conn.cursor() as cur:  # noqa: SLF001 — test cleanup only
            cur.execute("TRUNCATE pit_facts, pit_universe RESTART IDENTITY")
        store._conn.commit()
        yield store
    finally:
        store.close()


@pytest.fixture(params=["memory"])
def pit_store(request: pytest.FixtureRequest) -> PitStore:
    """Parametrize over in-memory (always) and postgres (when available)."""
    backend = request.param
    if backend == "memory":
        return InMemoryPitStore()
    if backend == "postgres":
        if not _postgres_available():
            pytest.skip(f"PostgreSQL not reachable via {TEST_PG_DSN_ENV_VAR}")
        store = PostgresPitStore.connect(postgres_test_dsn(), migrate=True)
        with store._conn.cursor() as cur:  # noqa: SLF001
            cur.execute("TRUNCATE pit_facts, pit_universe RESTART IDENTITY")
        store._conn.commit()
        request.addfinalizer(store.close)
        return store
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)


def _all_backends() -> list[str]:
    backends = ["memory"]
    if _postgres_available():
        backends.append("postgres")
    return backends


@pytest.fixture(params=_all_backends())
def all_pit_stores(request: pytest.FixtureRequest) -> PitStore:
    """Run leakage tests against every available backend."""
    if request.param == "memory":
        return InMemoryPitStore()
    store = PostgresPitStore.connect(postgres_test_dsn(), migrate=True)
    with store._conn.cursor() as cur:  # noqa: SLF001
        cur.execute("TRUNCATE pit_facts, pit_universe RESTART IDENTITY")
    store._conn.commit()
    request.addfinalizer(store.close)
    return store


@pytest.fixture
def seeded_ohlcv(pit_store: PitStore) -> dict[str, object]:
    """Seed one entity with five daily bars known as of KNOWLEDGE_INITIAL."""
    entity_id = "BIOTEST"
    records = ingest_synthetic_bars(
        pit_store,
        entity_id=entity_id,
        start_effective=T0,
        num_days=5,
        knowledge_time=KNOWLEDGE_INITIAL,
        seed=7,
    )
    return {
        "store": pit_store,
        "entity_id": entity_id,
        "records": records,
        "effective_range": (T0, T1),
        "as_of": T2,
    }
