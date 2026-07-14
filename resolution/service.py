"""Resolution writer service (C5.2).

Walks recorded questions, resolves the not-yet-resolved ones via a
:class:`~resolution.sources.ResolutionSource`, links each resolution to its
originating forecast (the latest for that question), and appends the immutable
``resolution`` record. Idempotent by construction: a question that already has a
resolution is skipped, so re-runs are safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.pit.models import ensure_utc
from core.registry.models import ResolutionInput
from core.registry.store import RegistryStore
from resolution.sources import ResolutionSource

__all__ = ["ResolutionRun", "ResolutionService"]


@dataclass(frozen=True)
class ResolutionRun:
    """Summary of one resolution pass."""

    resolved: tuple[str, ...]
    skipped: tuple[str, ...]


class ResolutionService:
    """Resolves open questions and writes their resolution records."""

    def __init__(self, *, store: RegistryStore, source: ResolutionSource) -> None:
        self._store = store
        self._source = source

    def resolve_open(self, *, since: datetime | None = None) -> ResolutionRun:
        """Resolve every not-yet-resolved question (optionally intook since ``since``)."""
        floor = ensure_utc(since) if since is not None else None
        resolved: list[str] = []
        skipped: list[str] = []
        for question in self._store.all_questions():
            question_id = question.question_id
            if floor is not None and question.knowledge_time < floor:
                skipped.append(question_id)
                continue
            if self._store.resolutions_for(question_id):
                skipped.append(question_id)  # already resolved -> idempotent skip
                continue
            outcome = self._source.resolve(question)
            if outcome is None:
                skipped.append(question_id)  # not yet resolvable
                continue
            forecasts = self._store.forecasts_for(question_id)
            forecast_id = forecasts[-1].forecast_id if forecasts else None
            resolution_id = self._store.record_resolution(
                ResolutionInput(
                    question_id=question_id,
                    resolved_value=outcome.resolved_value,
                    resolved_at=outcome.resolved_at,
                    source=outcome.source,
                    forecast_id=forecast_id,
                    resolved_label=outcome.resolved_label,
                    notes=outcome.notes,
                )
            )
            resolved.append(resolution_id)
        return ResolutionRun(resolved=tuple(resolved), skipped=tuple(skipped))
