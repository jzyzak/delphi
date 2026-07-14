"""ForecastBench adapter (C7.2).

Maps ForecastBench-style records into pinned :class:`BenchmarkQuestion` /
:class:`BenchmarkResolution` shapes. The network loader is out of scope for the
hermetic test suite; adapters are constructed from already-fetched records
(:meth:`from_records`) so mapping and as-of pinning are fully deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchmarks.base import BenchmarkQuestion, BenchmarkResolution, parse_dt

__all__ = ["ForecastBenchAdapter"]

_SOURCE = "forecastbench"


class ForecastBenchAdapter:
    """A ForecastBench question set built from fetched records."""

    def __init__(
        self,
        questions: Sequence[BenchmarkQuestion],
        resolutions: Sequence[BenchmarkResolution],
    ) -> None:
        self._questions = tuple(questions)
        self._resolutions = tuple(resolutions)

    @property
    def name(self) -> str:
        return _SOURCE

    def questions(self) -> Sequence[BenchmarkQuestion]:
        return self._questions

    def resolutions(self) -> Sequence[BenchmarkResolution]:
        return self._resolutions

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]]) -> ForecastBenchAdapter:
        """Map raw ForecastBench records into questions + resolutions.

        Each record needs ``id``, ``question`` (text), and ``as_of``. A record
        with a ``resolved_value`` + ``resolved_at`` also yields a resolution.
        """
        questions: list[BenchmarkQuestion] = []
        resolutions: list[BenchmarkResolution] = []
        for record in records:
            question = BenchmarkQuestion(
                source=_SOURCE,
                external_id=str(record["id"]),
                text=str(record["question"]),
                as_of=parse_dt(record["as_of"]),
                question_type=str(record.get("question_type", "binary")),
                domain=str(record.get("domain", "general")),
                resolution_criteria=str(record.get("resolution_criteria", "")),
                close_time=parse_dt(record["close_time"]) if record.get("close_time") else None,
                metadata={"benchmark": _SOURCE},
            )
            questions.append(question)
            if record.get("resolved_value") is not None and record.get("resolved_at"):
                resolutions.append(
                    BenchmarkResolution(
                        question_id=question.question_id,
                        resolved_value=float(record["resolved_value"]),
                        resolved_at=parse_dt(record["resolved_at"]),
                        source=str(record.get("resolution_source", _SOURCE)),
                    )
                )
        return cls(questions, resolutions)
