"""Metaculus adapter (C7.3).

Maps Metaculus-style question records into pinned benchmark shapes. Metaculus
questions carry a community prediction; when present it is retained in metadata
so a crowd-consensus baseline can be derived (see :mod:`benchmarks.market_consensus`).
Hermetic by construction: built from fetched records, no network in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchmarks.base import BenchmarkQuestion, BenchmarkResolution, parse_dt

__all__ = ["MetaculusAdapter"]

_SOURCE = "metaculus"


class MetaculusAdapter:
    """A Metaculus question set built from fetched records."""

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
    def from_records(cls, records: Sequence[dict[str, Any]]) -> MetaculusAdapter:
        """Map raw Metaculus records into questions + resolutions.

        Each record needs ``id``, ``title`` (text), and ``as_of``. ``community``
        (community prediction) is retained in metadata for the consensus baseline.
        A ``resolution`` (0/1) + ``resolved_at`` yields a resolution row.
        """
        questions: list[BenchmarkQuestion] = []
        resolutions: list[BenchmarkResolution] = []
        for record in records:
            metadata: dict[str, Any] = {"benchmark": _SOURCE}
            if record.get("community") is not None:
                metadata["community_prediction"] = float(record["community"])
            question = BenchmarkQuestion(
                source=_SOURCE,
                external_id=str(record["id"]),
                text=str(record["title"]),
                as_of=parse_dt(record["as_of"]),
                question_type=str(record.get("question_type", "binary")),
                domain=str(record.get("domain", "general")),
                resolution_criteria=str(record.get("resolution_criteria", "")),
                close_time=parse_dt(record["close_time"]) if record.get("close_time") else None,
                metadata=metadata,
            )
            questions.append(question)
            if record.get("resolution") is not None and record.get("resolved_at"):
                resolutions.append(
                    BenchmarkResolution(
                        question_id=question.question_id,
                        resolved_value=float(record["resolution"]),
                        resolved_at=parse_dt(record["resolved_at"]),
                        source=_SOURCE,
                    )
                )
        return cls(questions, resolutions)
