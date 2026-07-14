"""Reference adapter: synthetic daily OHLCV bars through the PIT write path.

Demonstrates revision handling and late-arriving (backfilled) data. No network
calls; all timestamps are injected by the caller.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from core.pit.models import FactRecord
from core.pit.store import PitStore

OHLCV_DATASET = "ohlcv_daily"


def _bar_values(
    rng: random.Random,
    *,
    base_price: float,
) -> dict[str, Any]:
    open_ = base_price + rng.uniform(-0.5, 0.5)
    close = open_ + rng.uniform(-1.0, 1.0)
    high = max(open_, close) + rng.uniform(0, 0.5)
    low = min(open_, close) - rng.uniform(0, 0.5)
    volume = rng.randint(100_000, 5_000_000)
    return {
        "open": round(open_, 4),
        "high": round(high, 4),
        "low": round(low, 4),
        "close": round(close, 4),
        "volume": volume,
    }


def ingest_synthetic_bars(
    store: PitStore,
    *,
    entity_id: str,
    start_effective: datetime,
    num_days: int,
    knowledge_time: datetime,
    seed: int = 42,
    base_price: float = 50.0,
) -> list[FactRecord]:
    """Ingest deterministic synthetic daily bars via the append-only write path.

    Contract: each bar's ``effective_time`` is the trading date; ``knowledge_time``
    is when the bar became known (injected, never read from wall-clock).

    Returns the appended ``FactRecord`` list for test assertions.
    """
    rng = random.Random(seed)
    records: list[FactRecord] = []
    price = base_price
    for day in range(num_days):
        effective = start_effective + timedelta(days=day)
        values = _bar_values(rng, base_price=price)
        price = float(values["close"])
        records.append(
            FactRecord(
                dataset=OHLCV_DATASET,
                entity_id=entity_id,
                effective_time=effective,
                knowledge_time=knowledge_time,
                values=values,
            )
        )
    store.append(records)
    return records


def append_bar_revision(
    store: PitStore,
    *,
    entity_id: str,
    effective_time: datetime,
    original: dict[str, Any],
    corrected_close: float,
    knowledge_time: datetime,
) -> FactRecord:
    """Append a correction for an existing bar (new row, later knowledge_time).

    Contract: the revision is invisible to ``as_of(T)`` when ``knowledge_time > T``;
    visible when ``knowledge_time <= T``.
    """
    revised = {**original, "close": corrected_close}
    if revised["high"] < corrected_close:
        revised["high"] = corrected_close
    if revised["low"] > corrected_close:
        revised["low"] = corrected_close
    record = FactRecord(
        dataset=OHLCV_DATASET,
        entity_id=entity_id,
        effective_time=effective_time,
        knowledge_time=knowledge_time,
        values=revised,
    )
    store.append([record])
    return record


def append_late_arrival_bar(
    store: PitStore,
    *,
    entity_id: str,
    effective_time: datetime,
    knowledge_time: datetime,
    seed: int = 99,
    base_price: float = 75.0,
) -> FactRecord:
    """Append a bar whose effective_time is in the past but knowledge_time is later.

    Contract: invisible to ``as_of(T)`` for any ``T < knowledge_time``; visible
    once ``as_of >= knowledge_time``.
    """
    rng = random.Random(seed)
    record = FactRecord(
        dataset=OHLCV_DATASET,
        entity_id=entity_id,
        effective_time=effective_time,
        knowledge_time=knowledge_time,
        values=_bar_values(rng, base_price=base_price),
    )
    store.append([record])
    return record


def utc_dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Convenience helper for tests and demos: tz-aware UTC datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)
