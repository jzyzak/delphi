"""Integration tests for PostgresJobStore against a real (test) database.

Opt-in via DELPHI_TEST_PG_DSN (postgres marker); hermetic per-test via
TRUNCATE. Proves the guarded transitions and ON CONFLICT idempotency hold
under the real engine, not just the mocked SQL contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from api.jobs import ForecastJob, JobStatus, PostgresJobStore
from tests.conftest import postgres_test_dsn

pytestmark = pytest.mark.postgres

T0 = datetime(2024, 6, 1, tzinfo=UTC)
T1 = datetime(2024, 6, 1, 0, 5, tzinfo=UTC)
PAYLOAD = {"question": "Will X ship?", "as_of": "2024-06-01T00:00:00+00:00"}


@pytest.fixture
def job_store() -> Iterator[PostgresJobStore]:
    dsn = postgres_test_dsn()
    store = PostgresJobStore.connect(dsn, migrate=True)
    try:
        with store._conn.cursor() as cur:  # noqa: SLF001 — test cleanup only
            cur.execute("TRUNCATE forecast_jobs")
        yield store
    finally:
        store.close()


def _job(job_id: str = "fj-pg-1", key: str | None = "key-1") -> ForecastJob:
    return ForecastJob(
        job_id=job_id,
        status=JobStatus.QUEUED,
        request=dict(PAYLOAD),
        idempotency_key=key,
        created_at=T0,
    )


def test_create_get_round_trip(job_store: PostgresJobStore) -> None:
    created_job, created = job_store.create(_job())
    assert created is True
    fetched = job_store.get(created_job.job_id)
    assert fetched is not None
    assert fetched.status is JobStatus.QUEUED
    assert fetched.request == PAYLOAD
    assert fetched.idempotency_key == "key-1"
    assert fetched.created_at == T0


def test_idempotency_key_conflict_returns_existing(job_store: PostgresJobStore) -> None:
    first, _ = job_store.create(_job("fj-pg-1"))
    second, created = job_store.create(_job("fj-pg-2"))
    assert created is False
    assert second.job_id == first.job_id
    assert job_store.get("fj-pg-2") is None


def test_keyless_jobs_do_not_conflict(job_store: PostgresJobStore) -> None:
    _, a = job_store.create(_job("fj-pg-1", key=None))
    _, b = job_store.create(_job("fj-pg-2", key=None))
    assert a and b


def test_claim_is_at_most_once(job_store: PostgresJobStore) -> None:
    job, _ = job_store.create(_job())
    claimed = job_store.claim(job.job_id, started_at=T0)
    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING
    assert claimed.started_at == T0
    assert job_store.claim(job.job_id, started_at=T1) is None


def test_complete_lifecycle(job_store: PostgresJobStore) -> None:
    job, _ = job_store.create(_job())
    job_store.claim(job.job_id, started_at=T0)
    job_store.complete(job.job_id, result={"probability": 0.6}, finished_at=T1)
    done = job_store.get(job.job_id)
    assert done is not None
    assert done.status is JobStatus.SUCCEEDED
    assert done.result == {"probability": 0.6}
    assert done.finished_at == T1
    # Terminal jobs are immutable: a late fail must not overwrite the result.
    job_store.fail(job.job_id, error="late", finished_at=T1)
    still_done = job_store.get(job.job_id)
    assert still_done is not None
    assert still_done.status is JobStatus.SUCCEEDED


def test_fail_from_queued(job_store: PostgresJobStore) -> None:
    job, _ = job_store.create(_job())
    job_store.fail(job.job_id, error="worker lost", finished_at=T1)
    failed = job_store.get(job.job_id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.error == "worker lost"


def test_complete_requires_running(job_store: PostgresJobStore) -> None:
    job, _ = job_store.create(_job())
    job_store.complete(job.job_id, result={"ok": True}, finished_at=T1)  # still queued
    fetched = job_store.get(job.job_id)
    assert fetched is not None
    assert fetched.status is JobStatus.QUEUED
    assert fetched.result is None
