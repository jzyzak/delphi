"""Hermeticity guards: postgres fixtures must be physically unable to hit prod.

Covers the ``postgres_test_dsn`` helper (skip / hard-fail / pass-through) and a
static scan proving no test reads ``DELPHI_PG_DSN`` from the environment — the
regression that let a frozen-clock fixture exhaust the production trials
ledger (§2.4) in 2026-07.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import TEST_PG_DSN_ENV_VAR, postgres_test_dsn

_TEST_DSN = "postgresql://postgres:pw@localhost:5433/delphi_test"
_PROD_DSN = "postgresql://postgres:pw@localhost:5433/delphi"


class TestPostgresTestDsn:
    def test_unset_skips(self) -> None:
        with pytest.raises(pytest.skip.Exception, match=TEST_PG_DSN_ENV_VAR):
            postgres_test_dsn({})

    def test_empty_skips(self) -> None:
        with pytest.raises(pytest.skip.Exception):
            postgres_test_dsn({TEST_PG_DSN_ENV_VAR: ""})

    def test_distinct_databases_pass_through(self) -> None:
        env = {TEST_PG_DSN_ENV_VAR: _TEST_DSN, "DELPHI_PG_DSN": _PROD_DSN}
        assert postgres_test_dsn(env) == _TEST_DSN

    def test_no_prod_dsn_passes_through(self) -> None:
        assert postgres_test_dsn({TEST_PG_DSN_ENV_VAR: _TEST_DSN}) == _TEST_DSN

    def test_same_database_fails_hard(self) -> None:
        env = {TEST_PG_DSN_ENV_VAR: _PROD_DSN, "DELPHI_PG_DSN": _PROD_DSN}
        with pytest.raises(pytest.fail.Exception, match="same database"):
            postgres_test_dsn(env)

    def test_same_database_spelled_differently_fails_hard(self) -> None:
        # Different credentials / parameter order, same (host, port, dbname).
        env = {
            TEST_PG_DSN_ENV_VAR: "postgresql://other:creds@localhost:5433/delphi",
            "DELPHI_PG_DSN": _PROD_DSN,
        }
        with pytest.raises(pytest.fail.Exception, match="same database"):
            postgres_test_dsn(env)

    def test_same_host_different_dbname_passes(self) -> None:
        env = {TEST_PG_DSN_ENV_VAR: _TEST_DSN, "DELPHI_PG_DSN": _PROD_DSN}
        assert postgres_test_dsn(env) == _TEST_DSN

    def test_unparseable_prod_dsn_falls_back_to_literal_compare(self) -> None:
        broken = "not a dsn at all :::"
        env = {TEST_PG_DSN_ENV_VAR: broken, "DELPHI_PG_DSN": f"  {broken} "}
        with pytest.raises(pytest.fail.Exception, match="same database"):
            postgres_test_dsn(env)

    def test_unparseable_but_different_passes(self) -> None:
        env = {TEST_PG_DSN_ENV_VAR: "not a dsn :::", "DELPHI_PG_DSN": "also broken ;;;"}
        assert postgres_test_dsn(env) == "not a dsn :::"


_ENV_READ = re.compile(r"os\.environ(?:\.get)?\s*[\[\(]\s*[\"']DELPHI_PG_DSN")
_ALLOWED = {"conftest.py", "test_hermeticity.py"}  # root-level helper + this file


def test_no_test_reads_prod_dsn_from_environment() -> None:
    """Structural non-regression: tests never read DELPHI_PG_DSN from the env.

    The live smoke suite (tests/live) is exempt by design — it probes real
    infrastructure read-only via settings, gated behind DELPHI_LIVE_SMOKE=1 —
    but even it must not read the raw env var directly.
    """
    tests_root = Path(__file__).parent
    offenders: list[str] = []
    for path in tests_root.rglob("*.py"):
        if path.parent == tests_root and path.name in _ALLOWED:
            continue
        if _ENV_READ.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(tests_root)))
    assert offenders == [], (
        f"tests reading DELPHI_PG_DSN from the environment: {offenders} — "
        "use tests.conftest.postgres_test_dsn (DELPHI_TEST_PG_DSN) instead."
    )
