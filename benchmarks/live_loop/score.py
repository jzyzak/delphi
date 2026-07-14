"""Live score job (C9.2).

Resolve matured questions, then score every resolved forecast on its realized
outcome. The resulting live metrics are untunable by construction (CLAUDE.md
§2.7): they are computed straight from resolved records, with no fitting, no
threshold, and only proper scores (§2.3). This is the number we publish.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from core.registry.store import RegistryStore
from evaluation.scoring import BrierScorer, LogScorer, ScoredRecord, mean_score
from resolution.service import ResolutionService

__all__ = ["LiveMetrics", "ScoreJob", "ScoreRun", "collect_scored_records"]


def collect_scored_records(store: RegistryStore) -> tuple[ScoredRecord, ...]:
    """Collect every resolved, binary (question, forecast) pair as a scored record."""
    records: list[ScoredRecord] = []
    for question in store.all_questions():
        forecasts = store.forecasts_for(question.question_id)
        resolutions = store.resolutions_for(question.question_id)
        if not forecasts or not resolutions:
            continue
        forecast = forecasts[-1]
        resolution = resolutions[-1]
        if forecast.probability is None or resolution.resolved_value not in (0.0, 1.0):
            continue
        records.append(
            ScoredRecord(
                question_id=question.question_id,
                domain=question.domain,
                probability=forecast.probability,
                outcome=resolution.resolved_value,
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class LiveMetrics:
    """Untunable live metrics over the resolved forecasts."""

    n: int
    brier: float | None
    log: float | None

    @classmethod
    def from_records(cls, records: Sequence[ScoredRecord]) -> LiveMetrics:
        if not records:
            return cls(n=0, brier=None, log=None)
        probs = [r.probability for r in records]
        outcomes = [r.outcome for r in records]
        return cls(
            n=len(records),
            brier=mean_score(BrierScorer(), probs, outcomes),
            log=mean_score(LogScorer(), probs, outcomes),
        )


@dataclass(frozen=True)
class ScoreRun:
    """Summary of one score pass: newly resolved ids + refreshed live metrics."""

    resolved: tuple[str, ...]
    metrics: LiveMetrics


class ScoreJob:
    """Resolves matured questions and recomputes the live metrics."""

    def __init__(self, *, store: RegistryStore, resolution_service: ResolutionService) -> None:
        self._store = store
        self._resolution = resolution_service

    def run(self, *, since: datetime | None = None) -> ScoreRun:
        """Resolve matured questions, then score all resolved forecasts."""
        resolution_run = self._resolution.resolve_open(since=since)
        metrics = LiveMetrics.from_records(collect_scored_records(self._store))
        return ScoreRun(resolved=resolution_run.resolved, metrics=metrics)
