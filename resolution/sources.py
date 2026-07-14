"""Resolution source adapters (C5.1).

A resolution source turns a closed :class:`~core.registry.models.Question` into a
ground-truth :class:`ResolvedOutcome` on the question's native scale, with source
provenance. Resolution is not forecast-forming, so ``resolved_at`` is supplied by
the source (never ``now()`` inside library code) and the criteria captured at
intake drive how the outcome is read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.pit.models import ensure_utc
from core.registry.models import Question

__all__ = [
    "MappingResolutionSource",
    "ResolutionSource",
    "ResolvedOutcome",
    "load_mapping_source",
    "provenance_source",
]


@dataclass(frozen=True)
class ResolvedOutcome:
    """A ground-truth outcome for a question, on its native scale."""

    resolved_value: float
    resolved_at: datetime
    source: str = ""
    resolved_label: str = ""
    notes: str = ""


@runtime_checkable
class ResolutionSource(Protocol):
    """Reads the criteria captured at intake and returns the outcome, if known."""

    def resolve(self, question: Question) -> ResolvedOutcome | None:
        """Return the outcome for ``question``, or ``None`` if not yet resolvable."""
        ...


def provenance_source(question: Question, fallback: str) -> str:
    """Derive a non-empty provenance label from the question's resolution sources."""
    if fallback.strip():
        return fallback
    sources = question.metadata.get("resolution_sources")
    if isinstance(sources, list) and sources:
        joined = "; ".join(str(s) for s in sources if str(s).strip())
        if joined:
            return joined
    return "unspecified"


class MappingResolutionSource:
    """Resolve from an explicit map of ``question_id -> outcome``.

    A deterministic adapter for backfills and tests: the outcome is supplied
    (e.g. read from an official result), and missing provenance is filled from
    the question's intake-captured resolution sources.
    """

    def __init__(self, outcomes: Mapping[str, ResolvedOutcome]) -> None:
        self._outcomes = dict(outcomes)

    def resolve(self, question: Question) -> ResolvedOutcome | None:
        outcome = self._outcomes.get(question.question_id)
        if outcome is None:
            return None
        return ResolvedOutcome(
            resolved_value=outcome.resolved_value,
            resolved_at=ensure_utc(outcome.resolved_at),
            source=provenance_source(question, outcome.source),
            resolved_label=outcome.resolved_label,
            notes=outcome.notes,
        )


def _parse_resolved_at(raw: Any) -> datetime:
    """Parse a ground-truth resolution timestamp (ISO-8601, trailing Z ok)."""
    if not isinstance(raw, str) or not raw.strip():
        msg = "each answer requires a 'resolved_at' ISO-8601 timestamp."
        raise ValueError(msg)
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def load_mapping_source(path: str | Path) -> MappingResolutionSource:
    """Load a JSON answer key into a :class:`MappingResolutionSource`.

    The file maps ``question_id -> {value, resolved_at, source?, label?, notes?}``,
    e.g.::

        {"q-abc": {"value": 1.0, "resolved_at": "2025-01-01T00:00:00Z",
                    "source": "official result", "label": "YES"}}

    This gives ``delphi resolve`` a real ground-truth source (before automated
    resolution adapters land) so the forecast -> resolve lifecycle is exercisable.
    Resolution is not forecast-forming, so ``resolved_at`` comes from the file,
    never ``now()`` (CLAUDE.md §2.1).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "answer key must be a JSON object mapping question_id -> outcome."
        raise ValueError(msg)
    outcomes: dict[str, ResolvedOutcome] = {}
    for question_id, record in data.items():
        if not isinstance(record, dict) or "value" not in record:
            msg = f"answer for {question_id!r} must be an object with a 'value'."
            raise ValueError(msg)
        outcomes[str(question_id)] = ResolvedOutcome(
            resolved_value=float(record["value"]),
            resolved_at=_parse_resolved_at(record.get("resolved_at")),
            source=str(record.get("source", "")),
            resolved_label=str(record.get("label", "")),
            notes=str(record.get("notes", "")),
        )
    return MappingResolutionSource(outcomes)
