"""Benchmark resolution source (C9.2 support).

Resolves a recorded registry :class:`~core.registry.models.Question` to its
ground-truth outcome using a benchmark's resolutions, joined by the benchmark
question id that the live harvest threaded into the question's metadata (see
:mod:`benchmarks.live_loop.harvest`). Resolution is not forecast-forming, so
``resolved_at`` comes from the benchmark, never ``now()`` (CLAUDE.md §2.1).
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.base import BenchmarkResolution
from core.registry.models import Question
from resolution.sources import ResolvedOutcome, provenance_source

__all__ = ["BENCHMARK_QUESTION_ID_KEY", "BenchmarkResolutionSource"]

# Metadata key under which the harvest records a question's benchmark id.
BENCHMARK_QUESTION_ID_KEY = "benchmark_question_id"


class BenchmarkResolutionSource:
    """Resolve registry questions from benchmark resolutions keyed by benchmark id.

    A question is resolvable iff it carries a benchmark id in its metadata and a
    matching resolution exists; otherwise ``resolve`` returns ``None`` and the
    :class:`~resolution.service.ResolutionService` leaves it open (idempotent).
    """

    def __init__(
        self,
        resolutions: Sequence[BenchmarkResolution],
        *,
        metadata_key: str = BENCHMARK_QUESTION_ID_KEY,
    ) -> None:
        self._by_id = {resolution.question_id: resolution for resolution in resolutions}
        self._metadata_key = metadata_key

    def resolve(self, question: Question) -> ResolvedOutcome | None:
        benchmark_id = question.metadata.get(self._metadata_key)
        if not isinstance(benchmark_id, str):
            return None
        resolution = self._by_id.get(benchmark_id)
        if resolution is None:
            return None
        return ResolvedOutcome(
            resolved_value=resolution.resolved_value,
            resolved_at=resolution.resolved_at,
            source=provenance_source(question, resolution.source),
            resolved_label=resolution.resolved_label,
        )
