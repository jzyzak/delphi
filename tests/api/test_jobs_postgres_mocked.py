"""Unit tests for PostgresJobStore using mocked connections (no DB).

Mirrors tests/registry/test_postgres_mocked.py: prove the SQL contract (guards,
ON CONFLICT idempotency, autocommit) without a live Postgres.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from api.jobs import ForecastJob, JobStatus, PostgresJobStore, _parse_jsonb

T0 = datetime(2024, 6, 1, tzinfo=UTC)
PAYLOAD = {"question": "Will X ship?", "as_of": "2024-06-01T00:00:00+00:00"}

_ROW: tuple[Any, ...] = (
    "fj-1",  # job_id
    "key-1",  # idempotency_key
    "succeeded",  # status
    dict(PAYLOAD),  # request
    {"object": "forecast.completion"},  # result
    None,  # error
    T0,  # created_at
    T0,  # started_at
    T0,  # finished_at
)


class TestParseJsonb:
    def test_parses_dict(self) -> None:
        assert _parse_jsonb({"a": 1}) == {"a": 1}

    def test_parses_json_string(self) -> None:
        assert _parse_jsonb('{"a": 1}') == {"a": 1}

    def test_rejects_unexpected_type(self) -> None:
        with pytest.raises(TypeError, match="Unexpected JSONB"):
            _parse_jsonb(42)

    def test_rejects_non_object_json(self) -> None:
        with pytest.raises(TypeError, match="Unexpected JSONB"):
            _parse_jsonb("[1, 2]")


def _store() -> tuple[PostgresJobStore, MagicMock, MagicMock]:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    return PostgresJobStore(conn), conn, cur


def _job(**overrides: Any) -> ForecastJob:
    defaults: dict[str, Any] = {
        "job_id": "fj-1",
        "status": JobStatus.QUEUED,
        "request": dict(PAYLOAD),
        "idempotency_key": "key-1",
        "created_at": T0,
    }
    defaults.update(overrides)
    return ForecastJob(**defaults)


class TestCreate:
    def test_insert_returns_created_true(self) -> None:
        store, _, cur = _store()
        cur.fetchone.return_value = ("fj-1",)
        job, created = store.create(_job())
        assert created is True
        assert job.job_id == "fj-1"
        sql_text = cur.execute.call_args_list[0].args[0]
        assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql_text
        params = cur.execute.call_args_list[0].args[1]
        assert params["request"] == json.dumps(PAYLOAD)
        assert params["status"] == "queued"

    def test_conflict_returns_existing_job(self) -> None:
        store, _, cur = _store()
        cur.fetchone.side_effect = [None, _ROW]  # insert skipped -> select existing
        job, created = store.create(_job())
        assert created is False
        assert job.status is JobStatus.SUCCEEDED
        assert job.result == {"object": "forecast.completion"}
        select_sql = cur.execute.call_args_list[1].args[0]
        assert "WHERE idempotency_key" in select_sql


class TestGet:
    def test_found_maps_row(self) -> None:
        store, _, cur = _store()
        cur.fetchone.return_value = _ROW
        job = store.get("fj-1")
        assert job is not None
        assert job.job_id == "fj-1"
        assert job.idempotency_key == "key-1"
        assert job.status is JobStatus.SUCCEEDED
        assert job.request == PAYLOAD
        assert job.created_at == T0

    def test_missing_returns_none(self) -> None:
        store, _, cur = _store()
        cur.fetchone.return_value = None
        assert store.get("fj-404") is None

    def test_jsonb_string_columns_are_parsed(self) -> None:
        store, _, cur = _store()
        row = list(_ROW)
        row[3] = json.dumps(PAYLOAD)
        row[4] = json.dumps({"ok": True})
        cur.fetchone.return_value = tuple(row)
        job = store.get("fj-1")
        assert job is not None
        assert job.request == PAYLOAD
        assert job.result == {"ok": True}


class TestClaim:
    def test_claim_returns_job_on_win(self) -> None:
        store, _, cur = _store()
        row = list(_ROW)
        row[2] = "running"
        cur.fetchone.return_value = tuple(row)
        job = store.claim("fj-1", started_at=T0)
        assert job is not None
        assert job.status is JobStatus.RUNNING
        sql_text = cur.execute.call_args.args[0]
        assert "AND status = %(queued)s" in sql_text
        params = cur.execute.call_args.args[1]
        assert params["queued"] == "queued"
        assert params["started_at"] == T0

    def test_claim_lost_returns_none(self) -> None:
        store, _, cur = _store()
        cur.fetchone.return_value = None
        assert store.claim("fj-1", started_at=T0) is None


class TestTerminalTransitions:
    def test_complete_guards_on_running(self) -> None:
        store, _, cur = _store()
        store.complete("fj-1", result={"ok": True}, finished_at=T0)
        sql_text = cur.execute.call_args.args[0]
        assert "AND status = %(running)s" in sql_text
        params = cur.execute.call_args.args[1]
        assert params["result"] == json.dumps({"ok": True})
        assert params["succeeded"] == "succeeded"

    def test_fail_guards_on_non_terminal(self) -> None:
        store, _, cur = _store()
        store.fail("fj-1", error="boom", finished_at=T0)
        sql_text = cur.execute.call_args.args[0]
        assert "status IN (%(queued)s, %(running)s)" in sql_text
        params = cur.execute.call_args.args[1]
        assert params["error"] == "boom"
        assert params["failed"] == "failed"


class TestConnection:
    @patch("api.jobs.psycopg.connect")
    @patch("api.jobs._MIGRATIONS_DIR")
    def test_connect_applies_migrations_with_autocommit(
        self, migrations_dir: MagicMock, connect: MagicMock
    ) -> None:
        conn = MagicMock()
        connect.return_value = conn
        migration = MagicMock()
        migration.read_text.return_value = "SELECT 1;"
        migrations_dir.glob.return_value = [migration]

        store = PostgresJobStore.connect("postgresql://test", migrate=True)
        assert isinstance(store, PostgresJobStore)
        assert connect.call_args.kwargs.get("autocommit") is True
        conn.transaction.assert_called()

    @patch("api.jobs.psycopg.connect")
    def test_connect_without_migrate_skips_migrations(self, connect: MagicMock) -> None:
        conn = MagicMock()
        connect.return_value = conn
        PostgresJobStore.connect("postgresql://test", migrate=False)
        conn.transaction.assert_not_called()

    def test_context_manager_closes(self) -> None:
        store, conn, _ = _store()
        with store:
            pass
        conn.close.assert_called_once()
