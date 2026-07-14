"""Atomic global trials-budget reservation against the ledger (06)."""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Postgres advisory-lock keys are signed bigint (int64); the key must fit in
# 63 bits to stay positive. "TRIALSBU" (8 bytes) fits; a 9-byte string would
# overflow into numeric and fail to match the bigint overload.
_BUDGET_ADVISORY_LOCK_KEY = 0x545249414C534255  # "TRIALSBU"


class ReservationStatus(StrEnum):
    """Lifecycle of a budget grant."""

    RESERVED = "reserved"
    COMMITTED = "committed"
    RELEASED = "released"


@dataclass(frozen=True)
class BudgetGrant:
    """Reservation of ``n`` trial slots against the firm-wide budget."""

    grant_id: str
    n: int


@dataclass(frozen=True)
class BudgetSnapshot:
    """Observed budget usage for tests and observability."""

    debited: int
    outstanding_reserved: int
    cap: int

    @property
    def headroom(self) -> int:
        return max(self.cap - self.debited - self.outstanding_reserved, 0)


class BudgetLedger(ABC):
    """Atomic reservation layer over the global trials ledger."""

    @abstractmethod
    def reserve_budget(self, *, n: int) -> BudgetGrant | None:
        """Atomically reserve ``n`` trials if the firm-wide budget allows."""

    @abstractmethod
    def commit(self, grant: BudgetGrant) -> None:
        """Mark a grant consumed after a successful agent run (idempotent)."""

    @abstractmethod
    def release(self, grant: BudgetGrant) -> None:
        """Return reserved capacity after failure or cancellation (idempotent)."""

    @abstractmethod
    def snapshot(self) -> BudgetSnapshot:
        """Current debited, outstanding-reserved, and cap values."""


class InMemoryBudgetLedger(BudgetLedger):
    """Lock-guarded in-memory budget ledger for tests and local development."""

    def __init__(
        self,
        cap: int,
        *,
        trials_count: Callable[[], int],
    ) -> None:
        if cap < 0:
            msg = "cap must be >= 0."
            raise ValueError(msg)
        self._cap = cap
        self._trials_count = trials_count
        self._lock = threading.Lock()
        self._grants: dict[str, tuple[int, ReservationStatus]] = {}

    def reserve_budget(self, *, n: int) -> BudgetGrant | None:
        if n <= 0:
            msg = "n must be > 0."
            raise ValueError(msg)
        with self._lock:
            snap = self._snapshot_unlocked()
            if snap.debited + snap.outstanding_reserved + n > self._cap:
                return None
            grant_id = str(uuid.uuid4())
            self._grants[grant_id] = (n, ReservationStatus.RESERVED)
            return BudgetGrant(grant_id=grant_id, n=n)

    def commit(self, grant: BudgetGrant) -> None:
        with self._lock:
            self._transition(grant.grant_id, ReservationStatus.COMMITTED)

    def release(self, grant: BudgetGrant) -> None:
        with self._lock:
            self._transition(grant.grant_id, ReservationStatus.RELEASED)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> BudgetSnapshot:
        outstanding = sum(
            n for n, status in self._grants.values() if status is ReservationStatus.RESERVED
        )
        return BudgetSnapshot(
            debited=self._trials_count(),
            outstanding_reserved=outstanding,
            cap=self._cap,
        )

    def _transition(self, grant_id: str, target: ReservationStatus) -> None:
        if grant_id not in self._grants:
            return
        n, _status = self._grants[grant_id]
        self._grants[grant_id] = (n, target)


class PostgresBudgetLedger(BudgetLedger):
    """PostgreSQL-backed atomic budget reservation."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        cap: int,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if cap < 0:
            msg = "cap must be >= 0."
            raise ValueError(msg)
        self._conn = conn
        self._cap = cap
        self._clock = clock or (lambda: datetime.now(UTC))
        # A psycopg connection is not safe for concurrent use by multiple
        # threads; serialize access so interleaved transactions cannot corrupt
        # session state. Cross-process serialization is handled by the advisory
        # lock inside the transaction.
        self._lock = threading.Lock()

    @classmethod
    def connect(
        cls,
        dsn: str,
        cap: int,
        *,
        migrate: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> PostgresBudgetLedger:
        """Open a connection and optionally apply pending migrations.

        autocommit=True so each reservation commits atomically via its own
        conn.transaction() block. With autocommit off, a prior read on the
        connection opens an implicit transaction that demotes the write's
        transaction() to an uncommitted savepoint (silent data loss on close).
        """
        conn = psycopg.connect(dsn, autocommit=True)
        ledger = cls(conn, cap, clock=clock)
        if migrate:
            ledger.apply_migrations()
        return ledger

    def apply_migrations(self) -> None:
        """Apply SQL migrations from ``orchestration/migrations/`` in sorted order."""
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        with self._conn.transaction(), self._conn.cursor() as cur:
            for path in migration_files:
                cur.execute(sql.SQL(path.read_text()))  # pyright: ignore[reportArgumentType]

    def reserve_budget(self, *, n: int) -> BudgetGrant | None:
        if n <= 0:
            msg = "n must be > 0."
            raise ValueError(msg)
        grant_id = str(uuid.uuid4())
        now = self._clock()
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(%(key)s)",
                {"key": _BUDGET_ADVISORY_LOCK_KEY},
            )
            cur.execute("SELECT COUNT(*) FROM trials_ledger")
            debited_row = cur.fetchone()
            debited = int(debited_row[0]) if debited_row else 0
            cur.execute(
                """
                SELECT COALESCE(SUM(n), 0)
                FROM budget_reservations
                WHERE status = %(status)s
                """,
                {"status": ReservationStatus.RESERVED.value},
            )
            outstanding_row = cur.fetchone()
            outstanding = int(outstanding_row[0]) if outstanding_row else 0
            if debited + outstanding + n > self._cap:
                return None
            cur.execute(
                """
                INSERT INTO budget_reservations (grant_id, n, status, reserved_at, updated_at)
                VALUES (%(grant_id)s, %(n)s, %(status)s, %(now)s, %(now)s)
                """,
                {
                    "grant_id": grant_id,
                    "n": n,
                    "status": ReservationStatus.RESERVED.value,
                    "now": now,
                },
            )
        return BudgetGrant(grant_id=grant_id, n=n)

    def commit(self, grant: BudgetGrant) -> None:
        """Mark the grant consumed and append its trials to the ledger (idempotent).

        The reserved -> committed transition is guarded, so a repeated commit finds
        no reserved row (``rowcount != 1``) and appends nothing. The ledger stays an
        accurate, append-only count (CLAUDE.md §2.4): ``grant.n`` rows are inserted
        the first time and never again for the same grant.
        """
        now = self._clock()
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE budget_reservations
                SET status = %(status)s, updated_at = %(now)s
                WHERE grant_id = %(grant_id)s
                  AND status = %(from_status)s
                """,
                {
                    "grant_id": grant.grant_id,
                    "status": ReservationStatus.COMMITTED.value,
                    "from_status": ReservationStatus.RESERVED.value,
                    "now": now,
                },
            )
            if cur.rowcount != 1:
                return  # already committed/released — never double-debit the ledger.
            cur.execute(
                """
                INSERT INTO trials_ledger (grant_id, recorded_at)
                SELECT %(grant_id)s, %(now)s
                FROM generate_series(1, %(n)s)
                """,
                {"grant_id": grant.grant_id, "now": now, "n": grant.n},
            )

    def release(self, grant: BudgetGrant) -> None:
        self._set_status(grant.grant_id, ReservationStatus.RELEASED)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trials_ledger")
            debited_row = cur.fetchone()
            debited = int(debited_row[0]) if debited_row else 0
            cur.execute(
                """
                SELECT COALESCE(SUM(n), 0)
                FROM budget_reservations
                WHERE status = %(status)s
                """,
                {"status": ReservationStatus.RESERVED.value},
            )
            outstanding_row = cur.fetchone()
            outstanding = int(outstanding_row[0]) if outstanding_row else 0
        return BudgetSnapshot(debited=debited, outstanding_reserved=outstanding, cap=self._cap)

    def _set_status(self, grant_id: str, status: ReservationStatus) -> None:
        now = self._clock()
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE budget_reservations
                SET status = %(status)s, updated_at = %(now)s
                WHERE grant_id = %(grant_id)s
                  AND status = %(from_status)s
                """,
                {
                    "grant_id": grant_id,
                    "status": status.value,
                    "from_status": ReservationStatus.RESERVED.value,
                    "now": now,
                },
            )

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> PostgresBudgetLedger:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
