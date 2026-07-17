"""Repo-wide test fixtures: hermetic Postgres DSN resolution.

Postgres-marked tests write fixture data (frozen clocks, TRUNCATEs) and must be
physically unable to touch the production database. They therefore run ONLY
against ``DELPHI_TEST_PG_DSN`` — never ``DELPHI_PG_DSN`` — and refuse to run at
all (hard fail, not skip) when the two resolve to the same database. This is
the structural fix for the 2026-07 trials-ledger pollution, where a fixture's
frozen 2025-01-01 clock wrote 100 committed trials into the real ledger and
exhausted the §2.4 budget.

The deliberate exception is ``tests/live`` — the operator smoke suite is
double-gated (``DELPHI_LIVE_SMOKE=1``) and exists to probe the *real*
infrastructure read-only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import pytest
from psycopg import conninfo

TEST_PG_DSN_ENV_VAR = "DELPHI_TEST_PG_DSN"
_PROD_DSN_ENV_VAR = "DELPHI_PG_DSN"


def _endpoint(dsn: str) -> tuple[str, str, str]:
    """Reduce a DSN to its (host, port, dbname) identity for comparison."""
    parts = conninfo.conninfo_to_dict(dsn)
    return (
        str(parts.get("host") or "localhost"),
        str(parts.get("port") or "5432"),
        str(parts.get("dbname") or ""),
    )


def postgres_test_dsn(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the dedicated TEST database DSN, or skip/fail.

    - ``DELPHI_TEST_PG_DSN`` unset -> ``pytest.skip`` (postgres tests are opt-in).
    - Test DSN resolves to the same (host, port, dbname) as ``DELPHI_PG_DSN``
      -> ``pytest.fail`` HARD: fixtures truncate tables and write frozen-clock
      rows, so pointing them at the production database corrupts it. A skip
      would hide the misconfiguration; failing surfaces it.
    """
    env = os.environ if environ is None else environ
    test_dsn = env.get(TEST_PG_DSN_ENV_VAR)
    if not test_dsn:
        pytest.skip(f"{TEST_PG_DSN_ENV_VAR} not set")
    prod_dsn = env.get(_PROD_DSN_ENV_VAR)
    if prod_dsn:
        try:
            same = _endpoint(test_dsn) == _endpoint(prod_dsn)
        except Exception:  # unparseable DSN: fall back to a literal comparison
            same = test_dsn.strip() == prod_dsn.strip()
        if same:
            pytest.fail(
                f"{TEST_PG_DSN_ENV_VAR} resolves to the same database as "
                f"{_PROD_DSN_ENV_VAR}. Postgres tests TRUNCATE tables and write "
                "fixture rows; they must use a dedicated test database "
                "(e.g. create `delphi_test` and point the test DSN at it)."
            )
    return test_dsn
