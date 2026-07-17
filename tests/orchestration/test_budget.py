"""Tests for the trials-budget ledger (CLAUDE.md §2.4).

The in-memory ledger is exercised directly; the Postgres ledger's append-only
``trials_ledger`` write on commit is verified against a dedicated TEST database
when ``DELPHI_TEST_PG_DSN`` is set (skipped otherwise). These fixtures TRUNCATE
tables and write frozen-clock rows, so they must never point at the production
database — ``postgres_test_dsn`` fails hard if they do.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core.orchestration.budget import (
    BudgetGrant,
    InMemoryBudgetLedger,
    PostgresBudgetLedger,
)
from tests.conftest import postgres_test_dsn
from tests.registry.conftest import IncrementingClock


class TestInMemoryBudgetLedger:
    def test_reserve_commit_release_cycle(self) -> None:
        ledger = InMemoryBudgetLedger(cap=10, trials_count=lambda: 0)
        grant = ledger.reserve_budget(n=3)
        assert grant is not None
        assert ledger.snapshot().outstanding_reserved == 3
        ledger.commit(grant)
        assert ledger.snapshot().outstanding_reserved == 0

    def test_release_returns_capacity(self) -> None:
        ledger = InMemoryBudgetLedger(cap=10, trials_count=lambda: 0)
        grant = ledger.reserve_budget(n=4)
        assert grant is not None
        ledger.release(grant)
        assert ledger.snapshot().outstanding_reserved == 0

    def test_fail_closed_when_over_cap(self) -> None:
        ledger = InMemoryBudgetLedger(cap=2, trials_count=lambda: 0)
        assert ledger.reserve_budget(n=3) is None

    def test_debited_counts_against_cap(self) -> None:
        ledger = InMemoryBudgetLedger(cap=5, trials_count=lambda: 4)
        assert ledger.reserve_budget(n=2) is None
        assert ledger.reserve_budget(n=1) is not None

    def test_negative_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="cap must be >= 0"):
            InMemoryBudgetLedger(cap=-1, trials_count=lambda: 0)

    def test_non_positive_n_raises(self) -> None:
        ledger = InMemoryBudgetLedger(cap=5, trials_count=lambda: 0)
        with pytest.raises(ValueError, match="n must be > 0"):
            ledger.reserve_budget(n=0)

    def test_commit_unknown_grant_is_noop(self) -> None:
        ledger = InMemoryBudgetLedger(cap=5, trials_count=lambda: 0)
        ledger.commit(BudgetGrant(grant_id="nope", n=1))  # no error
        assert ledger.snapshot().outstanding_reserved == 0

    def test_default_counts_own_committed_trials(self) -> None:
        # Without an injected trials_count, committed grants ARE the debit — a
        # fresh reserve must see prior spend, not zero (§2.4).
        ledger = InMemoryBudgetLedger(cap=5)
        grant = ledger.reserve_budget(n=3)
        assert grant is not None
        ledger.commit(grant)
        assert ledger.snapshot().debited == 3
        assert ledger.reserve_budget(n=3) is None
        assert ledger.reserve_budget(n=2) is not None

    def test_default_released_grants_do_not_debit(self) -> None:
        ledger = InMemoryBudgetLedger(cap=5)
        grant = ledger.reserve_budget(n=4)
        assert grant is not None
        ledger.release(grant)
        assert ledger.snapshot().debited == 0
        assert ledger.reserve_budget(n=5) is not None

    def test_injected_trials_count_overrides_internal(self) -> None:
        ledger = InMemoryBudgetLedger(cap=5, trials_count=lambda: 4)
        grant = ledger.reserve_budget(n=1)
        assert grant is not None
        ledger.commit(grant)
        # External count wins: internal committed grants are not added on top.
        assert ledger.snapshot().debited == 4

    def test_not_durable(self) -> None:
        assert InMemoryBudgetLedger(cap=1).durable is False

    def test_postgres_ledger_is_durable(self) -> None:
        from unittest.mock import MagicMock

        # Construction takes no network; durability is a class-level fact.
        assert PostgresBudgetLedger(MagicMock(), cap=1).durable is True


@pytest.fixture
def postgres_ledger() -> Iterator[PostgresBudgetLedger]:
    dsn = postgres_test_dsn()
    ledger = PostgresBudgetLedger.connect(dsn, cap=100, migrate=True, clock=IncrementingClock())
    try:
        with ledger._conn.cursor() as cur:  # noqa: SLF001 — test cleanup only
            cur.execute("TRUNCATE trials_ledger RESTART IDENTITY")
            cur.execute("TRUNCATE budget_reservations RESTART IDENTITY")
        yield ledger
    finally:
        ledger.close()


def _trials_count(ledger: PostgresBudgetLedger) -> int:
    with ledger._conn.cursor() as cur:  # noqa: SLF001 — test assertion only
        cur.execute("SELECT COUNT(*) FROM trials_ledger")
        row = cur.fetchone()
    return int(row[0]) if row else 0


@pytest.mark.postgres
class TestPostgresBudgetLedger:
    def test_commit_appends_n_trials(self, postgres_ledger: PostgresBudgetLedger) -> None:
        grant = postgres_ledger.reserve_budget(n=3)
        assert grant is not None
        assert _trials_count(postgres_ledger) == 0
        postgres_ledger.commit(grant)
        assert _trials_count(postgres_ledger) == 3
        assert postgres_ledger.snapshot().debited == 3
        assert postgres_ledger.snapshot().outstanding_reserved == 0

    def test_commit_is_idempotent(self, postgres_ledger: PostgresBudgetLedger) -> None:
        grant = postgres_ledger.reserve_budget(n=2)
        assert grant is not None
        postgres_ledger.commit(grant)
        postgres_ledger.commit(grant)  # second commit must not double-debit
        assert _trials_count(postgres_ledger) == 2

    def test_release_writes_no_trials(self, postgres_ledger: PostgresBudgetLedger) -> None:
        grant = postgres_ledger.reserve_budget(n=5)
        assert grant is not None
        postgres_ledger.release(grant)
        assert _trials_count(postgres_ledger) == 0
        assert postgres_ledger.snapshot().outstanding_reserved == 0

    def test_debited_trials_count_against_cap(self, postgres_ledger: PostgresBudgetLedger) -> None:
        grant = postgres_ledger.reserve_budget(n=99)
        assert grant is not None
        postgres_ledger.commit(grant)
        # 99 debited, cap 100 -> only 1 slot left.
        assert postgres_ledger.reserve_budget(n=2) is None
        assert postgres_ledger.reserve_budget(n=1) is not None
