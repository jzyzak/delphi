"""Generic as-of evidence read facade over the bitemporal PIT store.

Domain-agnostic (CLAUDE.md §5): this knows nothing about forecasts or any
particular domain. It is the single read path forecast-forming code uses to see evidence,
and it enforces Prime Directive §2.1 (NO LOOK-AHEAD): only facts whose
``knowledge_time <= as_of`` are ever returned. ``as_of`` is always an explicit
input — there is no ``now()`` here, and there must never be one.

The underlying :class:`~core.pit.store.PitStore.corpus_as_of` already filters by
knowledge-time structurally. This facade adds a typed, domain-neutral result
model, optional entity filtering / result capping, a deterministic ordering, and
a defensive post-read leakage assertion so any backend that ever violated the
contract fails loudly instead of leaking silently.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.pit.models import ensure_utc
from core.pit.store import PitStore

__all__ = [
    "EvidenceQuery",
    "EvidenceRecord",
    "LeakageError",
    "PitEvidenceView",
]


class LeakageError(RuntimeError):
    """Raised if the store surfaces a fact dated after the as-of ceiling.

    Defense-in-depth: ``corpus_as_of`` filters structurally, so this can only
    fire if a backend violates its contract. It exists so a leak is a loud
    failure, never a silent one (Prime Directive §2.1).
    """


class EvidenceQuery(BaseModel):
    """Parameters for an as-of evidence read.

    Contract: ``as_of`` is the knowledge-time ceiling. Only facts with
    ``knowledge_time <= as_of`` are visible; for each
    ``(dataset, entity_id, effective_time)`` the row with the greatest
    ``knowledge_time <= as_of`` is returned.
    """

    model_config = ConfigDict(frozen=True)

    datasets: tuple[str, ...]
    as_of: datetime
    effective_range: tuple[datetime, datetime] | None = None
    entity_ids: tuple[str, ...] | None = None
    limit: int | None = Field(default=None, ge=1)

    @field_validator("as_of", mode="before")
    @classmethod
    def _utc_as_of(cls, v: Any) -> datetime:
        return ensure_utc(v)

    @field_validator("effective_range", mode="before")
    @classmethod
    def _utc_range(cls, v: Any) -> tuple[datetime, datetime] | None:
        if v is None:
            return None
        start, end = v
        return (ensure_utc(start), ensure_utc(end))

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.datasets:
            msg = "EvidenceQuery.datasets must be non-empty."
            raise ValueError(msg)
        if self.effective_range is not None and self.effective_range[0] > self.effective_range[1]:
            msg = "EvidenceQuery.effective_range start must be <= end."
            raise ValueError(msg)
        return self


class EvidenceRecord(BaseModel):
    """One piece of evidence known as of the query ceiling.

    Domain-agnostic mirror of a bitemporal fact: ``values`` carries the
    payload (e.g. a snippet, a measurement) without any domain assumptions.
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


class PitEvidenceView:
    """As-of read facade over a :class:`~core.pit.store.PitStore`.

    The single sanctioned read path for evidence. Construct once with a store and
    call :meth:`evidence_as_of` with an explicit ceiling.
    """

    def __init__(self, store: PitStore) -> None:
        self._store = store

    def evidence_as_of(self, query: EvidenceQuery) -> tuple[EvidenceRecord, ...]:
        """Return evidence known as of ``query.as_of``, deterministically ordered.

        Rows are ordered by ``(knowledge_time, dataset, entity_id, effective_time)``
        for a stable, byte-reproducible result; ``limit`` (if set) keeps the first
        rows in that order. Any row dated after ``as_of`` raises
        :class:`LeakageError` — a contract violation, never a silent leak.
        """
        frame = self._store.corpus_as_of(
            datasets=query.datasets,
            as_of=query.as_of,
            effective_range=query.effective_range,
        )
        entity_filter = set(query.entity_ids) if query.entity_ids is not None else None
        records: list[EvidenceRecord] = []
        for row in frame.to_dicts():
            if entity_filter is not None and row["entity_id"] not in entity_filter:
                continue
            raw_values = row["values"] or {}
            # Polars unifies heterogeneous struct payloads with null fills; drop
            # null-valued keys so a payload round-trips as the caller stored it.
            values = {k: v for k, v in raw_values.items() if v is not None}
            record = EvidenceRecord(
                dataset=row["dataset"],
                entity_id=row["entity_id"],
                effective_time=row["effective_time"],
                knowledge_time=row["knowledge_time"],
                values=values,
            )
            if record.knowledge_time > query.as_of:
                msg = (
                    f"Leakage: store returned a fact dated {record.knowledge_time.isoformat()} "
                    f"after the as-of ceiling {query.as_of.isoformat()}."
                )
                raise LeakageError(msg)
            records.append(record)
        records.sort(key=lambda r: (r.knowledge_time, r.dataset, r.entity_id, r.effective_time))
        if query.limit is not None:
            records = records[: query.limit]
        return tuple(records)
