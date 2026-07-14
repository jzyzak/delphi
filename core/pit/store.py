"""Storage-agnostic PIT store interface and implementations.

Contract: all reads are pure functions of facts with ``knowledge_time <= as_of``.
Writes are append-only; no UPDATE or DELETE paths exist at the application layer.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import psycopg
from psycopg import sql

from core.pit.models import FactRecord, UniverseRecord, ensure_utc

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Canonical as-of output column order for deterministic, byte-comparable results.
_AS_OF_COLUMNS = ("entity_id", "effective_time", "knowledge_time", "values")
_CORPUS_COLUMNS = ("dataset", "entity_id", "effective_time", "knowledge_time", "values")


def _parse_jsonb_values(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    msg = f"Unexpected JSONB payload type: {type(raw)!r}"
    raise TypeError(msg)


def _corpus_to_frame(records: Sequence[FactRecord]) -> pl.DataFrame:
    """Convert corpus fact records to a polars frame including ``dataset``."""
    if not records:
        return pl.DataFrame(
            schema={
                "dataset": pl.Utf8,
                "entity_id": pl.Utf8,
                "effective_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "knowledge_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "values": pl.Struct([]),
            }
        )
    rows: list[dict[str, Any]] = [
        {
            "dataset": r.dataset,
            "entity_id": r.entity_id,
            "effective_time": r.effective_time,
            "knowledge_time": r.knowledge_time,
            "values": r.values,
        }
        for r in records
    ]
    frame = pl.DataFrame(rows).sort(["dataset", "entity_id", "effective_time"])
    return frame.select(list(_CORPUS_COLUMNS))


def _facts_to_frame(records: Sequence[FactRecord]) -> pl.DataFrame:
    """Convert fact records to the canonical as-of polars frame."""
    if not records:
        return pl.DataFrame(
            schema={
                "entity_id": pl.Utf8,
                "effective_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "knowledge_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "values": pl.Struct([]),
            }
        )
    rows: list[dict[str, Any]] = [
        {
            "entity_id": r.entity_id,
            "effective_time": r.effective_time,
            "knowledge_time": r.knowledge_time,
            "values": r.values,
        }
        for r in records
    ]
    frame = pl.DataFrame(rows).sort(["entity_id", "effective_time"])
    return frame.select(list(_AS_OF_COLUMNS))


def _select_as_of_facts(
    facts: Sequence[FactRecord],
    *,
    dataset: str,
    entity_ids: Sequence[str],
    effective_range: tuple[datetime, datetime],
    as_of: datetime,
) -> pl.DataFrame:
    """Core as-of selection: latest knowledge_time <= as_of per (entity, effective)."""
    as_of = ensure_utc(as_of)
    start, end = (ensure_utc(effective_range[0]), ensure_utc(effective_range[1]))
    entity_set = set(entity_ids)

    best: dict[tuple[str, datetime], FactRecord] = {}
    for fact in facts:
        if fact.dataset != dataset:
            continue
        if fact.entity_id not in entity_set:
            continue
        if not (start <= fact.effective_time <= end):
            continue
        if fact.knowledge_time > as_of:
            continue
        key = (fact.entity_id, fact.effective_time)
        current = best.get(key)
        if current is None or fact.knowledge_time > current.knowledge_time:
            best[key] = fact

    ordered = sorted(best.values(), key=lambda r: (r.entity_id, r.effective_time))
    return _facts_to_frame(ordered)


def _select_corpus_as_of(
    facts: Sequence[FactRecord],
    *,
    datasets: Sequence[str],
    effective_range: tuple[datetime, datetime] | None,
    as_of: datetime,
) -> pl.DataFrame:
    """Corpus scan: latest knowledge_time <= as_of per (dataset, entity, effective).

    Contract: rows with ``knowledge_time > as_of`` are skipped structurally during
    selection — never fetched and post-filtered.
    """
    as_of = ensure_utc(as_of)
    dataset_set = set(datasets)
    start: datetime | None = None
    end: datetime | None = None
    if effective_range is not None:
        start, end = (ensure_utc(effective_range[0]), ensure_utc(effective_range[1]))

    best: dict[tuple[str, str, datetime], FactRecord] = {}
    for fact in facts:
        if fact.dataset not in dataset_set:
            continue
        if fact.knowledge_time > as_of:
            continue
        if start is not None and end is not None and not (start <= fact.effective_time <= end):
            continue
        key = (fact.dataset, fact.entity_id, fact.effective_time)
        current = best.get(key)
        if current is None or fact.knowledge_time > current.knowledge_time:
            best[key] = fact

    ordered = sorted(
        best.values(),
        key=lambda r: (r.dataset, r.entity_id, r.effective_time),
    )
    return _corpus_to_frame(ordered)


def _select_universe_status(
    records: Sequence[UniverseRecord],
    *,
    universe: str,
    as_of: datetime,
) -> dict[str, UniverseRecord]:
    """Latest status record per entity known as of ``as_of``."""
    as_of = ensure_utc(as_of)
    best: dict[str, UniverseRecord] = {}
    for record in records:
        if record.universe != universe:
            continue
        if record.effective_time > as_of or record.knowledge_time > as_of:
            continue
        current = best.get(record.entity_id)
        if current is None or (record.effective_time, record.knowledge_time) > (
            current.effective_time,
            current.knowledge_time,
        ):
            best[record.entity_id] = record
    return best


class PitStore(ABC):
    """Append-only bitemporal store with as-of read semantics."""

    @abstractmethod
    def append(self, records: Sequence[FactRecord]) -> None:
        """Append fact records. Identical versions are idempotently ignored."""

    @abstractmethod
    def as_of(
        self,
        *,
        dataset: str,
        entity_ids: Sequence[str],
        effective_range: tuple[datetime, datetime],
        as_of: datetime,
    ) -> pl.DataFrame:
        """Return facts KNOWN as of ``as_of``.

        Contract: the result is a pure function of facts whose knowledge_time <= as_of.
        Inserting, revising, or removing any fact with knowledge_time > as_of MUST NOT
        change this result. For each (entity_id, effective_time) the row returned is the
        one with the greatest knowledge_time <= as_of.
        """

    @abstractmethod
    def corpus_as_of(
        self,
        *,
        datasets: Sequence[str],
        as_of: datetime,
        effective_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        """Return corpus facts KNOWN as of ``as_of`` across ``datasets``.

        Contract: only facts with ``knowledge_time <= as_of`` are visible. For each
        (dataset, entity_id, effective_time) the row returned is the one with the
        greatest ``knowledge_time <= as_of``. Inserting facts with later
        ``knowledge_time`` MUST NOT change this result.
        """

    @abstractmethod
    def append_universe(self, records: Sequence[UniverseRecord]) -> None:
        """Append universe membership/status records (append-only)."""

    @abstractmethod
    def universe_records(self, *, universe: str) -> tuple[UniverseRecord, ...]:
        """Return all universe records for internal selection (testing/diagnostics)."""


class InMemoryPitStore(PitStore):
    """In-memory reference implementation backed by Python collections."""

    def __init__(self) -> None:
        self._facts: list[FactRecord] = []
        self._universe: list[UniverseRecord] = []
        self._fact_keys: set[tuple[str, str, datetime, datetime]] = set()
        self._universe_keys: set[tuple[str, str, datetime, datetime]] = set()

    def append(self, records: Sequence[FactRecord]) -> None:
        for record in records:
            key = (
                record.dataset,
                record.entity_id,
                record.effective_time,
                record.knowledge_time,
            )
            if key in self._fact_keys:
                continue
            self._fact_keys.add(key)
            self._facts.append(record)

    def as_of(
        self,
        *,
        dataset: str,
        entity_ids: Sequence[str],
        effective_range: tuple[datetime, datetime],
        as_of: datetime,
    ) -> pl.DataFrame:
        return _select_as_of_facts(
            self._facts,
            dataset=dataset,
            entity_ids=entity_ids,
            effective_range=effective_range,
            as_of=as_of,
        )

    def corpus_as_of(
        self,
        *,
        datasets: Sequence[str],
        as_of: datetime,
        effective_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        return _select_corpus_as_of(
            self._facts,
            datasets=datasets,
            effective_range=effective_range,
            as_of=as_of,
        )

    def append_universe(self, records: Sequence[UniverseRecord]) -> None:
        for record in records:
            key = (
                record.universe,
                record.entity_id,
                record.effective_time,
                record.knowledge_time,
            )
            if key in self._universe_keys:
                continue
            self._universe_keys.add(key)
            self._universe.append(record)

    def universe_records(self, *, universe: str) -> tuple[UniverseRecord, ...]:
        return tuple(r for r in self._universe if r.universe == universe)

    @property
    def facts(self) -> tuple[FactRecord, ...]:
        """Exposed for tests that shuffle physical row order (L1)."""
        return tuple(self._facts)


class PostgresPitStore(PitStore):
    """PostgreSQL-backed PIT store using DISTINCT ON for as-of selection."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, dsn: str, *, migrate: bool = True) -> PostgresPitStore:
        """Open a connection and optionally apply pending migrations."""
        conn = psycopg.connect(dsn)
        store = cls(conn)
        if migrate:
            store.apply_migrations()
        return store

    def apply_migrations(self) -> None:
        """Apply SQL migrations from ``data/pit/migrations/`` in sorted order."""
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        with self._conn.cursor() as cur:
            for path in migration_files:
                cur.execute(sql.SQL(path.read_text()))  # pyright: ignore[reportArgumentType]
        self._conn.commit()

    def append(self, records: Sequence[FactRecord]) -> None:
        if not records:
            return
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO pit_facts (
                    dataset, entity_id, effective_time, knowledge_time, values
                )
                VALUES (
                    %(dataset)s, %(entity_id)s, %(effective_time)s,
                    %(knowledge_time)s, %(values)s
                )
                ON CONFLICT (dataset, entity_id, effective_time, knowledge_time) DO NOTHING
                """,
                [
                    {
                        "dataset": r.dataset,
                        "entity_id": r.entity_id,
                        "effective_time": r.effective_time,
                        "knowledge_time": r.knowledge_time,
                        "values": json.dumps(r.values),
                    }
                    for r in records
                ],
            )
        self._conn.commit()

    def as_of(
        self,
        *,
        dataset: str,
        entity_ids: Sequence[str],
        effective_range: tuple[datetime, datetime],
        as_of: datetime,
    ) -> pl.DataFrame:
        as_of = ensure_utc(as_of)
        start, end = (ensure_utc(effective_range[0]), ensure_utc(effective_range[1]))
        if not entity_ids:
            return _facts_to_frame([])

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (entity_id, effective_time)
                    entity_id,
                    effective_time,
                    knowledge_time,
                    values
                FROM pit_facts
                WHERE dataset = %(dataset)s
                  AND entity_id = ANY(%(entity_ids)s)
                  AND effective_time BETWEEN %(start)s AND %(end)s
                  AND knowledge_time <= %(as_of)s
                ORDER BY entity_id, effective_time, knowledge_time DESC
                """,
                {
                    "dataset": dataset,
                    "entity_ids": list(entity_ids),
                    "start": start,
                    "end": end,
                    "as_of": as_of,
                },
            )
            rows = cur.fetchall()

        records = [
            FactRecord(
                dataset=dataset,
                entity_id=row[0],
                effective_time=row[1],
                knowledge_time=row[2],
                values=_parse_jsonb_values(row[3]),
            )
            for row in rows
        ]
        return _facts_to_frame(records)

    def corpus_as_of(
        self,
        *,
        datasets: Sequence[str],
        as_of: datetime,
        effective_range: tuple[datetime, datetime] | None = None,
    ) -> pl.DataFrame:
        as_of = ensure_utc(as_of)
        if not datasets:
            return _corpus_to_frame([])

        params: dict[str, Any] = {
            "datasets": list(datasets),
            "as_of": as_of,
        }
        effective_clause = ""
        if effective_range is not None:
            start, end = (ensure_utc(effective_range[0]), ensure_utc(effective_range[1]))
            params["start"] = start
            params["end"] = end
            effective_clause = "AND effective_time BETWEEN %(start)s AND %(end)s"

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT ON (dataset, entity_id, effective_time)
                    dataset,
                    entity_id,
                    effective_time,
                    knowledge_time,
                    values
                FROM pit_facts
                WHERE dataset = ANY(%(datasets)s)
                  AND knowledge_time <= %(as_of)s
                  {effective_clause}
                ORDER BY dataset, entity_id, effective_time, knowledge_time DESC
                """,
                params,
            )
            rows = cur.fetchall()

        records = [
            FactRecord(
                dataset=row[0],
                entity_id=row[1],
                effective_time=row[2],
                knowledge_time=row[3],
                values=_parse_jsonb_values(row[4]),
            )
            for row in rows
        ]
        return _corpus_to_frame(records)

    def append_universe(self, records: Sequence[UniverseRecord]) -> None:
        if not records:
            return
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO pit_universe
                    (universe, entity_id, status, effective_time, knowledge_time, values)
                VALUES
                    (%(universe)s, %(entity_id)s, %(status)s,
                     %(effective_time)s, %(knowledge_time)s, %(values)s)
                ON CONFLICT (universe, entity_id, effective_time, knowledge_time) DO NOTHING
                """,
                [
                    {
                        "universe": r.universe,
                        "entity_id": r.entity_id,
                        "status": r.status,
                        "effective_time": r.effective_time,
                        "knowledge_time": r.knowledge_time,
                        "values": json.dumps(r.values),
                    }
                    for r in records
                ],
            )
        self._conn.commit()

    def universe_records(self, *, universe: str) -> tuple[UniverseRecord, ...]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT universe, entity_id, status, effective_time, knowledge_time, values
                FROM pit_universe
                WHERE universe = %(universe)s
                ORDER BY entity_id, effective_time, knowledge_time
                """,
                {"universe": universe},
            )
            rows = cur.fetchall()

        return tuple(
            UniverseRecord(
                universe=row[0],
                entity_id=row[1],
                status=row[2],
                effective_time=row[3],
                knowledge_time=row[4],
                values=_parse_jsonb_values(row[5]),
            )
            for row in rows
        )

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> PostgresPitStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
