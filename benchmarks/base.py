"""Benchmark adapter interface (C7.1).

An adapter turns an external question set into DELPHI's shapes with strict as-of
pinning: every :class:`BenchmarkQuestion` carries the ``as_of`` knowledge ceiling
the forecast must be formed under, and every :class:`BenchmarkResolution` carries
the ground truth known only after the question closed. The harness feed helper
joins model forecasts to resolutions into :class:`~evaluation.scoring.ScoredRecord`
rows — and refuses to build a row whose resolution predates its as-of ceiling
(the structural no-leakage guard, CLAUDE.md §2.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.pit.models import ensure_utc
from evaluation.scoring import ScoredRecord

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkQuestion",
    "BenchmarkResolution",
    "as_of_pins",
    "assert_no_leakage",
    "parse_dt",
    "scored_records",
]


def parse_dt(raw: Any) -> datetime:
    """Parse a required ISO-8601 date/datetime into a tz-aware UTC datetime."""
    if isinstance(raw, datetime):
        return ensure_utc(raw)
    if not isinstance(raw, str) or not raw.strip():
        msg = f"expected an ISO-8601 date string, got {raw!r}"
        raise ValueError(msg)
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


class BenchmarkQuestion(BaseModel):
    """An external question pinned to an as-of knowledge ceiling."""

    model_config = ConfigDict(frozen=True)

    source: str
    external_id: str
    text: str
    as_of: datetime
    question_type: str = "binary"
    domain: str = "general"
    resolution_criteria: str = ""
    close_time: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("as_of", "close_time")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return None if v is None else ensure_utc(v)

    @property
    def question_id(self) -> str:
        """Stable id joining questions to forecasts and resolutions."""
        return f"{self.source}:{self.external_id}"


class BenchmarkResolution(BaseModel):
    """Ground-truth outcome for a benchmark question (known post-close)."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    resolved_value: float
    resolved_at: datetime
    source: str = ""
    resolved_label: str = ""

    @field_validator("resolved_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """A named source of pinned questions and their (eventual) resolutions."""

    @property
    def name(self) -> str: ...

    def questions(self) -> Sequence[BenchmarkQuestion]:
        """Return the pinned benchmark questions."""
        ...

    def resolutions(self) -> Sequence[BenchmarkResolution]:
        """Return the known resolutions (may cover only some questions)."""
        ...


def as_of_pins(adapter: BenchmarkAdapter) -> dict[str, datetime]:
    """Map every question id to its as-of ceiling (the pin the chain must honor)."""
    return {q.question_id: q.as_of for q in adapter.questions()}


def assert_no_leakage(
    questions: Sequence[BenchmarkQuestion], resolutions: Sequence[BenchmarkResolution]
) -> None:
    """Assert each resolution is dated at/after its question's as-of ceiling.

    A resolution known *before* the as-of ceiling would mean the outcome was
    visible at forecast time — a §2.1 look-ahead violation baked into the data.
    """
    pins = {q.question_id: q.as_of for q in questions}
    for resolution in resolutions:
        pin = pins.get(resolution.question_id)
        if pin is None:
            continue
        if resolution.resolved_at < pin:
            msg = (
                f"leakage: resolution for {resolution.question_id} is dated "
                f"{resolution.resolved_at.isoformat()}, before its as-of ceiling "
                f"{pin.isoformat()}."
            )
            raise ValueError(msg)


def scored_records(
    forecasts: Mapping[str, float], adapter: BenchmarkAdapter
) -> tuple[ScoredRecord, ...]:
    """Join model forecasts to resolutions into scored records (harness feed).

    Only binary (0/1) resolutions are scored here; a question is included iff it
    has both a forecast and a resolution. The as-of no-leakage guard runs first.
    """
    questions = list(adapter.questions())
    resolutions = list(adapter.resolutions())
    assert_no_leakage(questions, resolutions)
    domain_of = {q.question_id: q.domain for q in questions}
    records: list[ScoredRecord] = []
    for resolution in resolutions:
        qid = resolution.question_id
        probability = forecasts.get(qid)
        if probability is None:
            continue
        if resolution.resolved_value not in (0.0, 1.0):
            continue  # non-binary outcomes are scored via CRPS elsewhere
        records.append(
            ScoredRecord(
                question_id=qid,
                domain=domain_of.get(qid, "general"),
                probability=probability,
                outcome=resolution.resolved_value,
            )
        )
    return tuple(records)
