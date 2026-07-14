"""Logged, budgeted holdout-access governor."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from core.registry.fingerprint import compute_record_hash

_HOLDOUT_STREAM_ID = "holdout_access_log"


class HoldoutBudgetExhausted(Exception):
    """Raised when the meta-layer attempts to exceed the holdout budget."""


@dataclass(frozen=True)
class HoldoutView:
    """Result of one logged, budgeted holdout access."""

    access_id: str
    payload: Mapping[str, Any]
    remaining_budget: int
    seq: int


@dataclass(frozen=True)
class HoldoutAccessRecord:
    """One tamper-evident holdout access event."""

    seq: int
    access_id: str
    reason: str
    prev_hash: str | None
    record_hash: str
    knowledge_time: datetime


@dataclass(frozen=True)
class HoldoutChainVerification:
    """Result of verifying the holdout access hash chain."""

    ok: bool
    broken_at_seq: int | None = None
    reason: str | None = None


class HoldoutSource(Protocol):
    """Injectable holdout payload seam (mocked in tests)."""

    def load(self) -> Mapping[str, Any]:
        """Return holdout metrics visible after a logged access."""
        ...


class HoldoutGovernor(ABC):
    """Governed holdout access: every peek is logged and debits budget."""

    @abstractmethod
    def access_holdout(self, *, reason: str) -> HoldoutView:
        """Sparing, logged, budgeted access to the locked holdout."""

    @abstractmethod
    def remaining_budget(self) -> int:
        """Remaining holdout accesses allowed."""

    @abstractmethod
    def verify_chain(self) -> HoldoutChainVerification:
        """Verify tamper-evidence of the holdout access log."""

    @abstractmethod
    def access_log(self) -> tuple[HoldoutAccessRecord, ...]:
        """Return the append-only access log (tests/audit)."""


class InMemoryHoldoutGovernor(HoldoutGovernor):
    """In-memory holdout governor with hash-chained access log."""

    def __init__(
        self,
        *,
        budget: int,
        source: HoldoutSource,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if budget < 0:
            msg = "budget must be >= 0."
            raise ValueError(msg)
        self._budget_cap = budget
        self._remaining = budget
        self._source = source
        self._clock = clock or (lambda: datetime.now(UTC))
        self._records: list[HoldoutAccessRecord] = []

    def access_holdout(self, *, reason: str) -> HoldoutView:
        if not reason.strip():
            msg = "reason must be non-empty."
            raise ValueError(msg)
        if self._remaining <= 0:
            msg = "Holdout budget exhausted; no unlogged path exists."
            raise HoldoutBudgetExhausted(msg)
        now = self._ensure_utc(self._clock())
        seq = len(self._records)
        access_id = f"holdout_{uuid.uuid4().hex}"
        prev_hash = self._records[-1].record_hash if self._records else None
        payload = {"access_id": access_id, "reason": reason}
        record_hash = compute_record_hash(
            stream_id=_HOLDOUT_STREAM_ID,
            seq=seq,
            record_kind="holdout_access",
            record_id=access_id,
            payload=payload,
            prev_hash=prev_hash,
            knowledge_time=now,
        )
        record = HoldoutAccessRecord(
            seq=seq,
            access_id=access_id,
            reason=reason,
            prev_hash=prev_hash,
            record_hash=record_hash,
            knowledge_time=now,
        )
        self._records.append(record)
        self._remaining -= 1
        return HoldoutView(
            access_id=access_id,
            payload=self._source.load(),
            remaining_budget=self._remaining,
            seq=seq,
        )

    def remaining_budget(self) -> int:
        return self._remaining

    def verify_chain(self) -> HoldoutChainVerification:
        prev_hash: str | None = None
        for index, record in enumerate(self._records):
            if record.seq != index:
                return HoldoutChainVerification(
                    ok=False,
                    broken_at_seq=record.seq,
                    reason=f"non-contiguous seq: expected {index}, found {record.seq}",
                )
            if record.prev_hash != prev_hash:
                return HoldoutChainVerification(
                    ok=False,
                    broken_at_seq=record.seq,
                    reason="prev_hash does not match prior record's hash",
                )
            payload = {"access_id": record.access_id, "reason": record.reason}
            recomputed = compute_record_hash(
                stream_id=_HOLDOUT_STREAM_ID,
                seq=record.seq,
                record_kind="holdout_access",
                record_id=record.access_id,
                payload=payload,
                prev_hash=record.prev_hash,
                knowledge_time=record.knowledge_time,
            )
            if recomputed != record.record_hash:
                return HoldoutChainVerification(
                    ok=False,
                    broken_at_seq=record.seq,
                    reason="record_hash mismatch",
                )
            prev_hash = record.record_hash
        return HoldoutChainVerification(ok=True)

    def access_log(self) -> tuple[HoldoutAccessRecord, ...]:
        return tuple(self._records)

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "clock must return tz-aware datetimes."
            raise ValueError(msg)
        return value.astimezone(UTC)


@dataclass(frozen=True)
class StaticHoldoutSource:
    """Deterministic holdout payload for tests."""

    payload: Mapping[str, Any]

    def load(self) -> Mapping[str, Any]:
        return self.payload
