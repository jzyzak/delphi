"""Async forecast jobs: submit now, poll for the result (C10.6).

Why this exists: hosted HTTP front-ends cap the total request time (AWS App
Runner hard-caps it at 120s, non-configurable), while a real DELPHI forecast
(agentic search + ensemble draws + leakage judge) routinely runs longer. The
synchronous ``POST /v1/forecast`` therefore cannot be served reliably behind
such a proxy. The job surface decouples the two: ``POST /v1/forecast/jobs``
validates and enqueues (fast), a small in-process worker pool executes the
*unchanged* forecast path (same registry writes, same leakage gate, same
explicit as-of ceiling), and ``GET /v1/forecast/jobs/{id}`` reports
status/result — optionally long-polling so each poll stays well under the
proxy's cap. Long-polling matters doubly on App Runner: instance CPU is
throttled to near-zero when no request is in flight, so an open poll is what
keeps the worker running at full speed.

Design notes:

- **No new infra** (CLAUDE.md §7). Execution is a ``ThreadPoolExecutor`` inside
  the API process; durability is the existing Postgres spine. An in-memory
  store backs local dev and tests (single process only — cross-process polls
  require Postgres, which production wiring uses whenever ``DELPHI_PG_DSN`` is
  set).
- **Idempotency.** A client-supplied ``idempotency_key`` maps to exactly one
  job, so a dashboard retrying a submit (e.g. after a network blip) can never
  pay for the same forecast twice. Resubmitting a key returns the existing job
  whatever its status.
- **At-most-one execution.** Workers *claim* a job with a guarded
  queued->running transition; when several processes see the same queued job
  (poll-driven revival after a restart, or a duplicate dispatch) exactly one
  claim wins and the others no-op.
- **Crash visibility.** A job whose worker died mid-run would stay ``running``
  forever; polls fail it after a configurable stale timeout so the client sees
  an honest terminal state instead of an eternal spinner. Queued jobs orphaned
  by a restart are re-dispatched by the next poll (safe via the claim guard).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import psycopg
from psycopg import sql
from pydantic import BaseModel

from api.compliance import ProviderOptOutError

__all__ = [
    "MAX_WAIT_S",
    "ForecastJob",
    "InMemoryJobStore",
    "JobExecutor",
    "JobManager",
    "JobStatus",
    "JobStore",
    "PostgresJobStore",
]

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Longest single long-poll a client may request. Stays comfortably under the
# App Runner 120s total-request cap (which includes connection setup and
# response write time).
MAX_WAIT_S = 90.0
_POLL_INTERVAL_S = 0.5

_STALE_ERROR = (
    "worker lost: the job exceeded the {timeout:g}s execution timeout without "
    "completing (instance restart or crash). Resubmit with a new idempotency key."
)


class JobStatus(StrEnum):
    """Lifecycle of one forecast job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED})


@dataclass(frozen=True)
class ForecastJob:
    """One persisted forecast job (request in, result or error out)."""

    job_id: str
    status: JobStatus
    request: dict[str, Any]
    idempotency_key: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore(ABC):
    """Durable job state with guarded (race-safe) status transitions."""

    @abstractmethod
    def create(self, job: ForecastJob) -> tuple[ForecastJob, bool]:
        """Insert ``job``; on an idempotency-key hit return the existing job.

        Returns ``(job, created)`` — ``created`` is False when the key already
        mapped to a job, in which case that existing job is returned unchanged.
        """

    @abstractmethod
    def get(self, job_id: str) -> ForecastJob | None:
        """Fetch a job by id, or ``None``."""

    @abstractmethod
    def claim(self, job_id: str, *, started_at: datetime) -> ForecastJob | None:
        """Atomically transition queued->running; ``None`` if not claimable.

        This is the at-most-one-execution guard: of N concurrent claimers
        exactly one receives the job.
        """

    @abstractmethod
    def complete(self, job_id: str, *, result: dict[str, Any], finished_at: datetime) -> None:
        """Transition running->succeeded with the result (no-op otherwise)."""

    @abstractmethod
    def fail(self, job_id: str, *, error: str, finished_at: datetime) -> None:
        """Transition a non-terminal job to failed (no-op on terminal jobs)."""


class InMemoryJobStore(JobStore):
    """Thread-safe in-memory store (tests + single-process local dev)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ForecastJob] = {}
        self._by_key: dict[str, str] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def create(self, job: ForecastJob) -> tuple[ForecastJob, bool]:
        with self._lock:
            key = job.idempotency_key
            if key is not None and key in self._by_key:
                return self._jobs[self._by_key[key]], False
            self._jobs[job.job_id] = job
            if key is not None:
                self._by_key[key] = job.job_id
            return job, True

    def get(self, job_id: str) -> ForecastJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def claim(self, job_id: str, *, started_at: datetime) -> ForecastJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                return None
            claimed = replace(job, status=JobStatus.RUNNING, started_at=started_at)
            self._jobs[job_id] = claimed
            return claimed

    def complete(self, job_id: str, *, result: dict[str, Any], finished_at: datetime) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.RUNNING:
                return
            self._jobs[job_id] = replace(
                job, status=JobStatus.SUCCEEDED, result=result, finished_at=finished_at
            )

    def fail(self, job_id: str, *, error: str, finished_at: datetime) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return
            self._jobs[job_id] = replace(
                job, status=JobStatus.FAILED, error=error, finished_at=finished_at
            )


_COLUMNS = (
    "job_id, idempotency_key, status, request, result, error, created_at, started_at, finished_at"
)


def _parse_jsonb(raw: object) -> dict[str, Any]:
    """Defensively decode a JSONB column (psycopg normally returns dicts)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    msg = f"Unexpected JSONB payload: {raw!r}"
    raise TypeError(msg)


class PostgresJobStore(JobStore):
    """PostgreSQL-backed job store (the production spine, CLAUDE.md §7).

    Cross-process/cross-instance visibility: any API worker can answer a poll
    for a job another worker is executing, and the guarded UPDATEs make claims
    and terminal transitions race-safe across all of them.
    """

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, dsn: str, *, migrate: bool = True) -> PostgresJobStore:
        # autocommit=True so each guarded transition commits atomically via its
        # own conn.transaction() block (same rationale as PostgresRunStateStore).
        conn = psycopg.connect(dsn, autocommit=True)
        store = cls(conn)
        if migrate:
            store.apply_migrations()
        return store

    def apply_migrations(self) -> None:
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        with self._conn.transaction(), self._conn.cursor() as cur:
            for path in migration_files:
                cur.execute(sql.SQL(path.read_text()))  # pyright: ignore[reportArgumentType]

    def create(self, job: ForecastJob) -> tuple[ForecastJob, bool]:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO forecast_jobs
                    (job_id, idempotency_key, status, request, result, error,
                     created_at, started_at, finished_at)
                VALUES
                    (%(job_id)s, %(key)s, %(status)s, %(request)s, NULL, NULL,
                     %(created_at)s, NULL, NULL)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING job_id
                """,
                {
                    "job_id": job.job_id,
                    "key": job.idempotency_key,
                    "status": job.status.value,
                    "request": json.dumps(job.request),
                    "created_at": job.created_at,
                },
            )
            inserted = cur.fetchone()
            if inserted is not None:
                return job, True
            cur.execute(
                f"SELECT {_COLUMNS} FROM forecast_jobs WHERE idempotency_key = %(key)s",
                {"key": job.idempotency_key},
            )
            row = cur.fetchone()
        if row is None:  # pragma: no cover - conflict implies the row exists
            msg = f"idempotency conflict for {job.idempotency_key!r} but no row found."
            raise RuntimeError(msg)
        return self._row_to_job(row), False

    def get(self, job_id: str) -> ForecastJob | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM forecast_jobs WHERE job_id = %(job_id)s",
                {"job_id": job_id},
            )
            row = cur.fetchone()
        return None if row is None else self._row_to_job(row)

    def claim(self, job_id: str, *, started_at: datetime) -> ForecastJob | None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE forecast_jobs
                SET status = %(running)s, started_at = %(started_at)s
                WHERE job_id = %(job_id)s AND status = %(queued)s
                RETURNING {_COLUMNS}
                """,
                {
                    "job_id": job_id,
                    "running": JobStatus.RUNNING.value,
                    "queued": JobStatus.QUEUED.value,
                    "started_at": started_at,
                },
            )
            row = cur.fetchone()
        return None if row is None else self._row_to_job(row)

    def complete(self, job_id: str, *, result: dict[str, Any], finished_at: datetime) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE forecast_jobs
                SET status = %(succeeded)s, result = %(result)s, finished_at = %(finished_at)s
                WHERE job_id = %(job_id)s AND status = %(running)s
                """,
                {
                    "job_id": job_id,
                    "succeeded": JobStatus.SUCCEEDED.value,
                    "running": JobStatus.RUNNING.value,
                    "result": json.dumps(result),
                    "finished_at": finished_at,
                },
            )

    def fail(self, job_id: str, *, error: str, finished_at: datetime) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE forecast_jobs
                SET status = %(failed)s, error = %(error)s, finished_at = %(finished_at)s
                WHERE job_id = %(job_id)s AND status IN (%(queued)s, %(running)s)
                """,
                {
                    "job_id": job_id,
                    "failed": JobStatus.FAILED.value,
                    "queued": JobStatus.QUEUED.value,
                    "running": JobStatus.RUNNING.value,
                    "error": error,
                    "finished_at": finished_at,
                },
            )

    @staticmethod
    def _row_to_job(row: tuple[Any, ...]) -> ForecastJob:
        return ForecastJob(
            job_id=row[0],
            idempotency_key=row[1],
            status=JobStatus(row[2]),
            request=_parse_jsonb(row[3]),
            result=None if row[4] is None else _parse_jsonb(row[4]),
            error=row[5],
            created_at=row[6],
            started_at=row[7],
            finished_at=row[8],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PostgresJobStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class JobExecutor(Protocol):
    """The slice of ``concurrent.futures.Executor`` the manager needs."""

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Any: ...

    def shutdown(self, wait: bool = True) -> None: ...


# The runner receives the stored request payload and returns the full API
# response model; the manager persists its ``model_dump``. Wiring builds it
# from ForecastService.forecast (see api.server.forecast_runner).
JobRunner = Callable[[Mapping[str, Any]], BaseModel]


class JobManager:
    """Owns the store + worker pool; the route-facing job operations."""

    def __init__(
        self,
        *,
        store: JobStore,
        runner: JobRunner,
        workers: int = 2,
        stale_after_s: float = 1800.0,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        id_factory: Callable[[], str] | None = None,
        executor: JobExecutor | None = None,
    ) -> None:
        self._store = store
        self._runner = runner
        self._workers = workers
        self._stale_after_s = stale_after_s
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or _default_sleep
        self._id_factory = id_factory or (lambda: f"fj-{uuid.uuid4().hex}")
        self._executor = executor
        self._executor_lock = threading.Lock()
        # Jobs dispatched by THIS process and not yet finished; prevents a poll
        # storm from queueing duplicate no-op claim attempts locally.
        self._inflight: set[str] = set()

    def submit(
        self, payload: Mapping[str, Any], *, idempotency_key: str | None = None
    ) -> tuple[ForecastJob, bool]:
        """Create (or idempotently find) a job and dispatch it if queued.

        Returns ``(job, created)``; ``created`` is False on an idempotency-key
        hit, in which case the existing job's current state is returned and
        nothing new is spent.
        """
        job = ForecastJob(
            job_id=self._id_factory(),
            status=JobStatus.QUEUED,
            request=dict(payload),
            idempotency_key=idempotency_key,
            created_at=self._clock(),
        )
        job, created = self._store.create(job)
        if job.status is JobStatus.QUEUED:
            self._dispatch(job.job_id)
            job = self._store.get(job.job_id) or job
        return job, created

    def get(self, job_id: str, *, wait_s: float = 0.0) -> ForecastJob | None:
        """Fetch a job, optionally long-polling until terminal or timeout.

        ``wait_s`` is clamped to [0, MAX_WAIT_S] (NaN counts as 0). A queued
        orphan (worker died before claiming) is re-dispatched; a running job
        past the stale timeout is failed so the client never polls forever.
        """
        if math.isnan(wait_s):
            wait_s = 0.0
        wait_s = max(0.0, min(wait_s, MAX_WAIT_S))
        deadline = self._clock() + timedelta(seconds=wait_s)
        while True:
            job = self._store.get(job_id)
            if job is None:
                return None
            if job.status is JobStatus.QUEUED:
                self._dispatch(job.job_id)
                job = self._store.get(job_id) or job
            if job.status is JobStatus.RUNNING and self._is_stale(job):
                self._store.fail(
                    job.job_id,
                    error=_STALE_ERROR.format(timeout=self._stale_after_s),
                    finished_at=self._clock(),
                )
                return self._store.get(job_id) or job
            if job.status in TERMINAL_STATUSES or self._clock() >= deadline:
                return job
            self._sleep(_POLL_INTERVAL_S)

    def close(self, *, wait: bool = True) -> None:
        """Shut down the worker pool (tests / graceful teardown)."""
        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait)

    def _is_stale(self, job: ForecastJob) -> bool:
        anchor = job.started_at or job.created_at
        if anchor is None:
            return False
        return self._clock() - anchor > timedelta(seconds=self._stale_after_s)

    def _dispatch(self, job_id: str) -> None:
        with self._executor_lock:
            if job_id in self._inflight:
                return
            self._inflight.add(job_id)
            executor = self._ensure_executor()
        executor.submit(self._run, job_id)

    def _ensure_executor(self) -> JobExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="delphi-job"
            )
        return self._executor

    def _run(self, job_id: str) -> None:
        try:
            job = self._store.claim(job_id, started_at=self._clock())
            if job is None:  # another worker won the claim, or it was failed
                return
            try:
                result = self._runner(job.request)
            except ProviderOptOutError as exc:
                self._store.fail(
                    job_id, error=f"provider_opt_out: {exc}", finished_at=self._clock()
                )
            except ValueError as exc:  # includes pydantic ValidationError
                self._store.fail(job_id, error=f"invalid_request: {exc}", finished_at=self._clock())
            except Exception as exc:
                logger.exception("forecast job %s failed", job_id)
                self._store.fail(job_id, error=f"forecast_failed: {exc}", finished_at=self._clock())
            else:
                self._store.complete(
                    job_id, result=result.model_dump(mode="json"), finished_at=self._clock()
                )
        finally:
            with self._executor_lock:
                self._inflight.discard(job_id)


def _default_sleep(seconds: float) -> None:  # pragma: no cover - trivial wall-clock wrapper
    import time

    time.sleep(seconds)
