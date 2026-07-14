"""Unit and property tests for the PIT data layer."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from core.pit.adapters.fixtures import OHLCV_DATASET, ingest_synthetic_bars, utc_dt
from core.pit.models import AsOfQuery, FactRecord, UniverseQuery, UniverseRecord, ensure_utc
from core.pit.store import InMemoryPitStore, PitStore, _select_as_of_facts, _select_universe_status

T0 = utc_dt(2024, 1, 1)
T1 = utc_dt(2024, 1, 5)
KNOWLEDGE = utc_dt(2024, 1, 6)


class TestModels:
    def test_rejects_naive_datetime_on_fact_record(self) -> None:
        naive = datetime(2024, 1, 1, 12, 0)
        with pytest.raises(ValueError, match="Naive datetimes"):
            FactRecord(
                dataset="x",
                entity_id="E",
                effective_time=naive,
                knowledge_time=naive,
                values={},
            )

    def test_rejects_knowledge_before_effective(self) -> None:
        with pytest.raises(ValueError, match="knowledge_time must be"):
            FactRecord(
                dataset="x",
                entity_id="E",
                effective_time=utc_dt(2024, 1, 5),
                knowledge_time=utc_dt(2024, 1, 1),
                values={},
            )

    def test_normalizes_non_utc_to_utc(self) -> None:
        from datetime import timezone

        eastern = datetime(2024, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
        record = FactRecord(
            dataset="x",
            entity_id="E",
            effective_time=eastern,
            knowledge_time=eastern,
            values={},
        )
        assert record.effective_time.tzinfo == UTC

    def test_as_of_query_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="effective_range start"):
            AsOfQuery(
                dataset="x",
                entity_ids=("E",),
                effective_range=(utc_dt(2024, 2, 1), utc_dt(2024, 1, 1)),
                as_of=utc_dt(2024, 3, 1),
            )

    def test_universe_query_accepts_utc(self) -> None:
        q = UniverseQuery(universe="u", as_of=utc_dt(2024, 1, 1))
        assert q.as_of.tzinfo == UTC

    def test_as_of_query_accepts_entity_tuple(self) -> None:
        q = AsOfQuery(
            dataset="d",
            entity_ids=("A", "B"),
            effective_range=(utc_dt(2024, 1, 1), utc_dt(2024, 1, 31)),
            as_of=utc_dt(2024, 2, 1),
        )
        assert q.entity_ids == ("A", "B")


class TestStore:
    def test_empty_as_of_returns_empty_frame(self, memory_store: InMemoryPitStore) -> None:
        frame = memory_store.as_of(
            dataset=OHLCV_DATASET,
            entity_ids=["MISSING"],
            effective_range=(T0, T1),
            as_of=KNOWLEDGE,
        )
        assert frame.is_empty()
        assert frame.columns == ["entity_id", "effective_time", "knowledge_time", "values"]

    def test_effective_range_is_inclusive(self, memory_store: InMemoryPitStore) -> None:
        ingest_synthetic_bars(
            memory_store,
            entity_id="E",
            start_effective=T0,
            num_days=3,
            knowledge_time=KNOWLEDGE,
        )
        end = T0 + timedelta(days=2)
        frame = memory_store.as_of(
            dataset=OHLCV_DATASET,
            entity_ids=["E"],
            effective_range=(T0, end),
            as_of=KNOWLEDGE,
        )
        assert frame.height == 3

    def test_latest_knowledge_wins(self, memory_store: InMemoryPitStore) -> None:
        entity_id = "E"
        memory_store.append(
            [
                FactRecord(
                    dataset=OHLCV_DATASET,
                    entity_id=entity_id,
                    effective_time=T0,
                    knowledge_time=utc_dt(2024, 1, 2),
                    values={"open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 100},
                ),
                FactRecord(
                    dataset=OHLCV_DATASET,
                    entity_id=entity_id,
                    effective_time=T0,
                    knowledge_time=utc_dt(2024, 1, 4),
                    values={"open": 1, "high": 1, "low": 1, "close": 2.0, "volume": 100},
                ),
            ]
        )
        frame = memory_store.as_of(
            dataset=OHLCV_DATASET,
            entity_ids=[entity_id],
            effective_range=(T0, T0),
            as_of=utc_dt(2024, 1, 10),
        )
        assert frame["values"].struct.field("close")[0] == 2.0

    def test_idempotent_append(self, memory_store: InMemoryPitStore) -> None:
        record = FactRecord(
            dataset=OHLCV_DATASET,
            entity_id="E",
            effective_time=T0,
            knowledge_time=KNOWLEDGE,
            values={"open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1},
        )
        memory_store.append([record, record])
        assert len(memory_store.facts) == 1

    def test_no_mutate_or_delete_methods_on_store(self) -> None:
        forbidden = {"update", "delete", "remove", "overwrite", "upsert"}
        for cls in (InMemoryPitStore, PitStore):
            if cls is PitStore:
                # ABC may not define these; check concrete impls primarily.
                continue
            public = {
                name
                for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
                if not name.startswith("_")
            }
            assert forbidden.isdisjoint(public)

    def test_ensure_utc_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="Naive"):
            ensure_utc(datetime(2024, 1, 1))


class TestUniverse:
    def test_empty_universe(self, memory_store: InMemoryPitStore) -> None:
        latest = _select_universe_status([], universe="empty", as_of=KNOWLEDGE)
        assert latest == {}

    def test_latest_status_by_effective_and_knowledge(self, memory_store: InMemoryPitStore) -> None:
        memory_store.append_universe(
            [
                UniverseRecord(
                    universe="u",
                    entity_id="A",
                    status="active",
                    effective_time=utc_dt(2024, 1, 1),
                    knowledge_time=utc_dt(2024, 1, 1),
                ),
                UniverseRecord(
                    universe="u",
                    entity_id="A",
                    status="withdrawn",
                    effective_time=utc_dt(2024, 1, 1),
                    knowledge_time=utc_dt(2024, 1, 5),
                ),
            ]
        )
        latest = _select_universe_status(
            memory_store.universe_records(universe="u"),
            universe="u",
            as_of=utc_dt(2024, 1, 10),
        )
        assert latest["A"].status == "withdrawn"


class TestFixturesAdapter:
    def test_synthetic_bars_are_deterministic(self, memory_store: InMemoryPitStore) -> None:
        r1 = ingest_synthetic_bars(
            memory_store,
            entity_id="D",
            start_effective=T0,
            num_days=3,
            knowledge_time=KNOWLEDGE,
            seed=99,
        )
        store2 = InMemoryPitStore()
        r2 = ingest_synthetic_bars(
            store2,
            entity_id="D",
            start_effective=T0,
            num_days=3,
            knowledge_time=KNOWLEDGE,
            seed=99,
        )
        assert [x.values for x in r1] == [x.values for x in r2]


# Hypothesis strategies for property tests
_utc_datetimes = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2026, 1, 1),
    timezones=st.just(UTC),
)


@settings(max_examples=50, deadline=None)
@given(
    as_of=_utc_datetimes,
    future_offset_days=st.integers(min_value=1, max_value=365),
)
def test_as_of_monotonicity_under_future_appends(as_of: datetime, future_offset_days: int) -> None:
    """Adding facts with knowledge_time > as_of must never change as_of(as_of)."""
    assume(as_of >= KNOWLEDGE)
    store = InMemoryPitStore()
    ingest_synthetic_bars(
        store,
        entity_id="P",
        start_effective=T0,
        num_days=3,
        knowledge_time=KNOWLEDGE,
    )
    baseline = store.as_of(
        dataset=OHLCV_DATASET,
        entity_ids=["P"],
        effective_range=(T0, T0 + timedelta(days=2)),
        as_of=as_of,
    )
    future_kt = as_of + timedelta(days=future_offset_days)
    store.append(
        [
            FactRecord(
                dataset=OHLCV_DATASET,
                entity_id="P",
                effective_time=T0,
                knowledge_time=future_kt,
                values={"open": 0, "high": 0, "low": 0, "close": 999.0, "volume": 1},
            )
        ]
    )
    after = store.as_of(
        dataset=OHLCV_DATASET,
        entity_ids=["P"],
        effective_range=(T0, T0 + timedelta(days=2)),
        as_of=as_of,
    )
    assert baseline.equals(after)


@settings(max_examples=30, deadline=None)
@given(knowledge_time=_utc_datetimes)
def test_select_as_of_facts_matches_store(knowledge_time: datetime) -> None:
    """Internal selector agrees with store for a single fact."""
    if knowledge_time < T0:
        knowledge_time = T0
    fact = FactRecord(
        dataset=OHLCV_DATASET,
        entity_id="H",
        effective_time=T0,
        knowledge_time=knowledge_time,
        values={"open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1},
    )
    store = InMemoryPitStore()
    store.append([fact])
    direct = _select_as_of_facts(
        [fact],
        dataset=OHLCV_DATASET,
        entity_ids=["H"],
        effective_range=(T0, T0),
        as_of=knowledge_time + timedelta(days=1),
    )
    via_store = store.as_of(
        dataset=OHLCV_DATASET,
        entity_ids=["H"],
        effective_range=(T0, T0),
        as_of=knowledge_time + timedelta(days=1),
    )
    assert direct.equals(via_store)
