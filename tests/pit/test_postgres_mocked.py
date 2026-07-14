"""Unit tests for PostgresPitStore using mocked connections (no DB required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.pit.adapters.fixtures import OHLCV_DATASET, utc_dt
from core.pit.models import FactRecord, UniverseRecord
from core.pit.store import PostgresPitStore, _parse_jsonb_values

T0 = utc_dt(2024, 1, 1)
KNOWLEDGE = utc_dt(2024, 1, 6)


class TestParseJsonb:
    def test_parses_dict(self) -> None:
        assert _parse_jsonb_values({"a": 1}) == {"a": 1}

    def test_parses_json_string(self) -> None:
        assert _parse_jsonb_values('{"close": 1.5}') == {"close": 1.5}

    def test_rejects_unexpected_type(self) -> None:
        with pytest.raises(TypeError, match="Unexpected JSONB"):
            _parse_jsonb_values(42)


class TestPostgresPitStoreMocked:
    def _mock_store(self) -> tuple[PostgresPitStore, MagicMock]:
        conn = MagicMock()
        store = PostgresPitStore(conn)
        return store, conn

    def test_append_noop_on_empty(self) -> None:
        store, conn = self._mock_store()
        store.append([])
        conn.cursor.assert_not_called()

    def test_append_commits(self) -> None:
        store, conn = self._mock_store()
        record = FactRecord(
            dataset=OHLCV_DATASET,
            entity_id="E",
            effective_time=T0,
            knowledge_time=KNOWLEDGE,
            values={"open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1},
        )
        store.append([record])
        conn.cursor.return_value.__enter__.return_value.executemany.assert_called_once()
        conn.commit.assert_called()

    def test_as_of_empty_entity_ids(self) -> None:
        store, _conn = self._mock_store()
        frame = store.as_of(
            dataset=OHLCV_DATASET,
            entity_ids=[],
            effective_range=(T0, T0),
            as_of=KNOWLEDGE,
        )
        assert frame.is_empty()

    def test_as_of_maps_rows(self) -> None:
        store, conn = self._mock_store()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [
            ("E", T0, KNOWLEDGE, {"close": 2.0}),
        ]
        frame = store.as_of(
            dataset=OHLCV_DATASET,
            entity_ids=["E"],
            effective_range=(T0, T0),
            as_of=KNOWLEDGE,
        )
        assert frame.height == 1
        assert frame["values"].struct.field("close")[0] == 2.0

    def test_append_universe_and_query(self) -> None:
        store, conn = self._mock_store()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [
            ("u", "A", "active", T0, KNOWLEDGE, {}),
        ]
        store.append_universe(
            [
                UniverseRecord(
                    universe="u",
                    entity_id="A",
                    status="active",
                    effective_time=T0,
                    knowledge_time=KNOWLEDGE,
                )
            ]
        )
        records = store.universe_records(universe="u")
        assert len(records) == 1
        assert records[0].entity_id == "A"

    def test_context_manager_closes(self) -> None:
        store, conn = self._mock_store()
        with store:
            pass
        conn.close.assert_called_once()

    @patch("core.pit.store.psycopg.connect")
    @patch("core.pit.store._MIGRATIONS_DIR")
    def test_connect_applies_migrations(
        self, migrations_dir: MagicMock, connect: MagicMock
    ) -> None:
        conn = MagicMock()
        connect.return_value = conn
        migration = MagicMock()
        migration.read_text.return_value = "SELECT 1;"
        migrations_dir.glob.return_value = [migration]

        store = PostgresPitStore.connect("postgresql://test", migrate=True)
        assert isinstance(store, PostgresPitStore)
        conn.cursor.return_value.__enter__.return_value.execute.assert_called()
        conn.commit.assert_called()
