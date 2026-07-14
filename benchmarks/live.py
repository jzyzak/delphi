"""Live-harvest adapter (C7.5).

Harvests genuinely-open questions and pins each to ``as_of = harvest_time`` — the
forecast is formed on questions whose answers do not yet exist, which is what
makes the live number untunable and publishable (CLAUDE.md §2.7). Already-forecast
questions are deduped against a provided set of seen ids so re-harvests are safe.
Live questions have no resolutions yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from benchmarks.base import BenchmarkQuestion, BenchmarkResolution, parse_dt
from core.pit.models import ensure_utc

__all__ = ["LiveHarvestAdapter"]

_SOURCE = "live"


class LiveHarvestAdapter:
    """Open questions pinned to harvest time, deduped against seen ids."""

    def __init__(self, questions: Sequence[BenchmarkQuestion]) -> None:
        self._questions = tuple(questions)

    @property
    def name(self) -> str:
        return _SOURCE

    def questions(self) -> Sequence[BenchmarkQuestion]:
        return self._questions

    def resolutions(self) -> Sequence[BenchmarkResolution]:
        return ()  # open questions have not resolved yet

    @classmethod
    def harvest(
        cls,
        records: Sequence[dict[str, Any]],
        *,
        harvest_time: datetime,
        seen_ids: Iterable[str] = (),
    ) -> LiveHarvestAdapter:
        """Pin open questions to ``harvest_time`` and drop already-seen ones.

        ``harvest_time`` is an explicit input (never ``now()`` inside library
        code, §2.1); it becomes each question's as-of ceiling.
        """
        pin = ensure_utc(harvest_time)
        already = set(seen_ids)
        questions: list[BenchmarkQuestion] = []
        for record in records:
            external_id = str(record["id"])
            question_id = f"{_SOURCE}:{external_id}"
            if question_id in already:
                continue  # dedupe: already forecast in a prior harvest
            questions.append(
                BenchmarkQuestion(
                    source=_SOURCE,
                    external_id=external_id,
                    text=str(record["question"]),
                    as_of=pin,
                    question_type=str(record.get("question_type", "binary")),
                    domain=str(record.get("domain", "general")),
                    resolution_criteria=str(record.get("resolution_criteria", "")),
                    close_time=parse_dt(record["close_time"]) if record.get("close_time") else None,
                    metadata={"benchmark": _SOURCE, "harvest_time": pin.isoformat()},
                )
            )
        return cls(questions)
