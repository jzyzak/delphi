"""Restart-safe orchestration run-state persistence."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from core.orchestration.types import ClaimResult, LoopName, StepStatus

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


@dataclass(frozen=True)
class StepRecord:
    """One persisted orchestration step."""

    step_id: str
    loop_name: LoopName
    tick_at: datetime
    status: StepStatus
    error_message: str | None = None


class RunStateStore(ABC):
    """Idempotent, restart-safe step claim and completion tracking."""

    @abstractmethod
    def claim_step(
        self,
        *,
        step_id: str,
        loop_name: LoopName,
        tick_at: datetime,
    ) -> ClaimResult:
        """Claim a step for execution, or skip if already succeeded."""

    @abstractmethod
    def mark_succeeded(self, *, step_id: str) -> None:
        """Record successful completion (idempotent)."""

    @abstractmethod
    def mark_failed(self, *, step_id: str, error_message: str) -> None:
        """Record failure so a restart may retry (idempotent)."""

    @abstractmethod
    def get_step(self, step_id: str) -> StepRecord | None:
        """Fetch a step record if present."""


class InMemoryRunStateStore(RunStateStore):
    """In-memory run-state store for tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: dict[str, StepRecord] = {}

    def claim_step(
        self,
        *,
        step_id: str,
        loop_name: LoopName,
        tick_at: datetime,
    ) -> ClaimResult:
        with self._lock:
            existing = self._steps.get(step_id)
            if existing is not None:
                if existing.status is StepStatus.SUCCEEDED:
                    return ClaimResult.ALREADY_SUCCEEDED
                self._steps[step_id] = StepRecord(
                    step_id=step_id,
                    loop_name=loop_name,
                    tick_at=tick_at,
                    status=StepStatus.RUNNING,
                )
                return ClaimResult.CLAIMED
            self._steps[step_id] = StepRecord(
                step_id=step_id,
                loop_name=loop_name,
                tick_at=tick_at,
                status=StepStatus.RUNNING,
            )
            return ClaimResult.CLAIMED

    def mark_succeeded(self, *, step_id: str) -> None:
        with self._lock:
            record = self._steps.get(step_id)
            if record is None:
                return
            self._steps[step_id] = StepRecord(
                step_id=record.step_id,
                loop_name=record.loop_name,
                tick_at=record.tick_at,
                status=StepStatus.SUCCEEDED,
            )

    def mark_failed(self, *, step_id: str, error_message: str) -> None:
        with self._lock:
            record = self._steps.get(step_id)
            if record is None:
                return
            self._steps[step_id] = StepRecord(
                step_id=record.step_id,
                loop_name=record.loop_name,
                tick_at=record.tick_at,
                status=StepStatus.FAILED,
                error_message=error_message,
            )

    def get_step(self, step_id: str) -> StepRecord | None:
        with self._lock:
            return self._steps.get(step_id)


class PostgresRunStateStore(RunStateStore):
    """PostgreSQL-backed run-state store."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._conn = conn
        self._clock = clock or (lambda: datetime.now(UTC))

    @classmethod
    def connect(
        cls,
        dsn: str,
        *,
        migrate: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> PostgresRunStateStore:
        # autocommit=True so each claim/update commits atomically via its own
        # conn.transaction() block. With autocommit off, a prior read on the
        # connection opens an implicit transaction that demotes the write's
        # transaction() to an uncommitted savepoint (silent data loss on close).
        conn = psycopg.connect(dsn, autocommit=True)
        store = cls(conn, clock=clock)
        if migrate:
            store.apply_migrations()
        return store

    def apply_migrations(self) -> None:
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        with self._conn.transaction(), self._conn.cursor() as cur:
            for path in migration_files:
                cur.execute(sql.SQL(path.read_text()))  # pyright: ignore[reportArgumentType]

    def claim_step(
        self,
        *,
        step_id: str,
        loop_name: LoopName,
        tick_at: datetime,
    ) -> ClaimResult:
        now = self._clock()
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO orchestration_runs (step_id, loop_name, tick_at, status, started_at)
                VALUES (%(step_id)s, %(loop_name)s, %(tick_at)s, %(status)s, %(now)s)
                ON CONFLICT (step_id) DO NOTHING
                RETURNING step_id
                """,
                {
                    "step_id": step_id,
                    "loop_name": loop_name.value,
                    "tick_at": tick_at,
                    "status": StepStatus.RUNNING.value,
                    "now": now,
                },
            )
            inserted = cur.fetchone()
            if inserted is not None:
                return ClaimResult.CLAIMED
            cur.execute(
                "SELECT status FROM orchestration_runs WHERE step_id = %(step_id)s",
                {"step_id": step_id},
            )
            row = cur.fetchone()
            if row is None:
                return ClaimResult.CLAIMED
            status = StepStatus(row[0])
            if status is StepStatus.SUCCEEDED:
                return ClaimResult.ALREADY_SUCCEEDED
            cur.execute(
                """
                UPDATE orchestration_runs
                SET status = %(status)s,
                    started_at = %(now)s,
                    finished_at = NULL,
                    error_message = NULL
                WHERE step_id = %(step_id)s
                  AND status <> %(succeeded)s
                """,
                {
                    "step_id": step_id,
                    "status": StepStatus.RUNNING.value,
                    "now": now,
                    "succeeded": StepStatus.SUCCEEDED.value,
                },
            )
        return ClaimResult.CLAIMED

    def mark_succeeded(self, *, step_id: str) -> None:
        now = self._clock()
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE orchestration_runs
                SET status = %(status)s, finished_at = %(now)s, error_message = NULL
                WHERE step_id = %(step_id)s
                """,
                {
                    "step_id": step_id,
                    "status": StepStatus.SUCCEEDED.value,
                    "now": now,
                },
            )

    def mark_failed(self, *, step_id: str, error_message: str) -> None:
        now = self._clock()
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE orchestration_runs
                SET status = %(status)s, finished_at = %(now)s, error_message = %(error)s
                WHERE step_id = %(step_id)s
                """,
                {
                    "step_id": step_id,
                    "status": StepStatus.FAILED.value,
                    "now": now,
                    "error": error_message,
                },
            )

    def get_step(self, step_id: str) -> StepRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT step_id, loop_name, tick_at, status, error_message
                FROM orchestration_runs
                WHERE step_id = %(step_id)s
                """,
                {"step_id": step_id},
            )
            row = cur.fetchone()
        if row is None:
            return None
        return StepRecord(
            step_id=row[0],
            loop_name=LoopName(row[1]),
            tick_at=row[2],
            status=StepStatus(row[3]),
            error_message=row[4],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PostgresRunStateStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
