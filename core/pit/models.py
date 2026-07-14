"""Typed domain models for the point-in-time data layer.

All timestamps are tz-aware UTC. Naive datetimes are rejected at the boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Dataset identifiers for market-data facts (stored in pit_facts.dataset).
OHLCV_DATASET = "ohlcv_daily"
TICKER_MAP_DATASET = "security.ticker_map"
CLASSIFICATION_DATASET = "security.classification"
CORPORATE_ACTIONS_DATASET = "corporate_actions"
SHARES_OUTSTANDING_DATASET = "fundamentals.shares_outstanding"
US_LISTINGS_UNIVERSE = "us_listings"


def ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes and normalize to UTC.

    Contract: every timestamp entering or leaving the PIT layer is tz-aware UTC.
    """
    if dt.tzinfo is None:
        msg = "Naive datetimes are not allowed; provide a tz-aware UTC datetime."
        raise ValueError(msg)
    return dt.astimezone(UTC)


class FactRecord(BaseModel):
    """A single append-only bitemporal fact.

    Contract: ``effective_time`` is when the fact is true in the world;
    ``knowledge_time`` is when we could first have known it. Corrections are
    new rows with a later ``knowledge_time``, never in-place updates.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    entity_id: str
    effective_time: datetime
    knowledge_time: datetime
    values: dict[str, Any]

    @field_validator("effective_time", "knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _knowledge_not_before_effective(self) -> Self:
        if self.knowledge_time < self.effective_time:
            msg = (
                "knowledge_time must be >= effective_time "
                "(cannot know a fact before it is true in the world)."
            )
            raise ValueError(msg)
        return self


class AsOfQuery(BaseModel):
    """Parameters for an as-of read against a dataset.

    Contract: the query ceiling is ``as_of`` (knowledge_time). Only facts with
    ``knowledge_time <= as_of`` are visible; for each (entity_id, effective_time)
    the row with the greatest knowledge_time <= as_of is returned.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    entity_ids: tuple[str, ...]
    effective_range: tuple[datetime, datetime]
    as_of: datetime

    @field_validator("as_of", mode="before")
    @classmethod
    def _utc_as_of(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @field_validator("effective_range", mode="before")
    @classmethod
    def _utc_range(cls, v: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
        start, end = v
        return (ensure_utc(start), ensure_utc(end))

    @model_validator(mode="after")
    def _range_ordered(self) -> Self:
        if self.effective_range[0] > self.effective_range[1]:
            msg = "effective_range start must be <= end."
            raise ValueError(msg)
        return self


UniverseStatus = Literal["active", "withdrawn"]


class UniverseRecord(BaseModel):
    """Bitemporal universe membership / status record (append-only).

    Contract: ``effective_time`` is when the status change takes effect;
    ``knowledge_time`` is when we learned of it. An entity is in the
    point-in-time universe at T iff its latest status record (by effective_time,
    then knowledge_time, both <= T) has status ``active``.
    """

    model_config = ConfigDict(frozen=True)

    universe: str
    entity_id: str
    status: UniverseStatus
    effective_time: datetime
    knowledge_time: datetime
    values: dict[str, Any] = {}

    @field_validator("effective_time", "knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class UniverseQuery(BaseModel):
    """Parameters for a point-in-time universe query.

    Contract: returns entities whose latest known status as of ``as_of`` is
    ``active``, with no survivorship bias.
    """

    model_config = ConfigDict(frozen=True)

    universe: str
    as_of: datetime

    @field_validator("as_of", mode="before")
    @classmethod
    def _utc_as_of(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class ListingFilter(BaseModel):
    """Optional basic listing filter applied as of the query ceiling.

    Contract: uses only the latest bar known as of ``as_of``; never future data.
    """

    model_config = ConfigDict(frozen=True)

    min_price: float | None = Field(default=None, gt=0)
    min_volume: int | None = Field(default=None, ge=0)


CorporateActionType = Literal["split", "dividend"]


class CorporateAction(BaseModel):
    """Normalized corporate action before ingestion into the PIT store.

    Contract: ``announce_date`` becomes both effective_time and knowledge_time on
    the stored fact; ``ex_date`` lives in the payload and governs price adjustment.
    """

    model_config = ConfigDict(frozen=True)

    permanent_id: str
    action_type: CorporateActionType
    announce_date: datetime
    ex_date: datetime
    split_ratio: float | None = None
    cash_amount: float | None = None
    currency: str = "USD"

    @field_validator("announce_date", "ex_date")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        if self.action_type == "split" and (self.split_ratio is None or self.split_ratio <= 0):
            msg = "split actions require split_ratio > 0."
            raise ValueError(msg)
        if self.action_type == "dividend" and (self.cash_amount is None or self.cash_amount < 0):
            msg = "dividend actions require cash_amount >= 0."
            raise ValueError(msg)
        return self

    def to_fact_values(self) -> dict[str, Any]:
        """Serialize to the JSONB payload stored in pit_facts."""
        return {
            "type": self.action_type,
            "ex_date": self.ex_date.isoformat(),
            "split_ratio": self.split_ratio,
            "cash_amount": self.cash_amount,
            "currency": self.currency,
        }
