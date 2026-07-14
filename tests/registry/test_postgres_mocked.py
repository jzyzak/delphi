"""Unit tests for PostgresRegistryStore using mocked connections (no DB)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from core.registry.store import PostgresRegistryStore, _parse_jsonb

KT = datetime(2025, 1, 1, tzinfo=UTC)


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
            _parse_jsonb("[1, 2, 3]")


class TestPersistMocked:
    def _store(self) -> tuple[PostgresRegistryStore, MagicMock]:
        conn = MagicMock()
        store = PostgresRegistryStore(conn, clock=lambda: KT)
        return store, conn

    def test_first_append_uses_seq_zero_and_null_prev(self) -> None:
        store, conn = self._store()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = None

        event = store._persist(  # noqa: SLF001
            stream_id="exp_1",
            stream_kind="experiment",
            record_kind="experiment",
            record_id="exp_1",
            payload={"experiment_id": "exp_1"},
        )

        assert event.seq == 0
        assert event.prev_hash is None
        assert event.knowledge_time == KT
        # advisory lock + select + insert
        assert cur.execute.call_count == 3

    def test_subsequent_append_chains_on_prior(self) -> None:
        store, conn = self._store()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = (0, "prior-hash")

        event = store._persist(  # noqa: SLF001
            stream_id="exp_1",
            stream_kind="experiment",
            record_kind="result",
            record_id="res_1",
            payload={"result_id": "res_1"},
        )

        assert event.seq == 1
        assert event.prev_hash == "prior-hash"

    def test_row_to_event_maps_columns(self) -> None:
        store, _conn = self._store()
        row = (
            "exp_1",
            "experiment",
            0,
            "experiment",
            "exp_1",
            {"experiment_id": "exp_1"},
            None,
            "hash-0",
            KT,
        )
        event = store._row_to_event(row)  # noqa: SLF001
        assert event.stream_id == "exp_1"
        assert event.payload == {"experiment_id": "exp_1"}
        assert event.knowledge_time == KT

    def test_context_manager_closes(self) -> None:
        store, conn = self._store()
        with store:
            pass
        conn.close.assert_called_once()

    @patch("core.registry.store.psycopg.connect")
    @patch("core.registry.store._MIGRATIONS_DIR")
    def test_connect_applies_migrations(
        self, migrations_dir: MagicMock, connect: MagicMock
    ) -> None:
        conn = MagicMock()
        connect.return_value = conn
        migration = MagicMock()
        migration.read_text.return_value = "SELECT 1;"
        migrations_dir.glob.return_value = [migration]

        store = PostgresRegistryStore.connect("postgresql://test", migrate=True)
        assert isinstance(store, PostgresRegistryStore)
        # The connection must be autocommit so each append commits atomically via
        # its own conn.transaction() block (guards the silent-drop regression).
        assert connect.call_args.kwargs.get("autocommit") is True
        # Migrations run inside a single transaction (not a bare commit()).
        conn.transaction.assert_called()
