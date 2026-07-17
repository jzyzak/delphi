"""Live harvest job (C9.1).

Pull genuinely-open questions, forecast each via the conductor (pinned to the
harvest time — never ``now()`` inside forecast code, §2.1), and persist them as
pending forecasts in the registry. A pending forecast is simply one without a
resolution yet; the score job (C9.2) closes the loop once the question matures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.base import BenchmarkAdapter
from conductor.corpus import CorpusWriter
from conductor.heuristic import HeuristicConductor
from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

__all__ = ["HarvestJob", "HarvestRun"]


@dataclass(frozen=True)
class HarvestRun:
    """Summary of one harvest pass."""

    pending: tuple[str, ...]
    refused: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.pending)


class HarvestJob:
    """Forecasts open questions via the conductor, persisting them as pending."""

    def __init__(
        self,
        *,
        conductor: HeuristicConductor,
        corpus_writer: CorpusWriter | None = None,
    ) -> None:
        self._conductor = conductor
        self._corpus_writer = corpus_writer

    def run(self, adapter: BenchmarkAdapter) -> HarvestRun:
        """Forecast every open question in ``adapter`` at its harvest-time pin.

        The benchmark question id is threaded into the recorded question's
        metadata so the score job can later resolve it against the benchmark
        (see :class:`resolution.benchmark_source.BenchmarkResolutionSource`).
        """
        pending: list[str] = []
        refused: list[str] = []
        questions: Sequence = adapter.questions()
        for question in questions:
            metadata: dict[str, object] = {
                BENCHMARK_QUESTION_ID_KEY: question.question_id,
                "benchmark_source": question.source,
                "benchmark_external_id": question.external_id,
            }
            freeze = question.metadata.get(
                "freeze_value", question.metadata.get("community_prediction")
            )
            if freeze is not None:
                metadata["market_freeze_value"] = float(freeze)
            if question.close_time is not None:
                # Gives the series-threshold estimator its horizon.
                metadata["resolution_date"] = question.close_time.isoformat()
            result = self._conductor.conduct(question.text, as_of=question.as_of, metadata=metadata)
            forecast = result.forecast
            if forecast.accepted and forecast.question_id is not None:
                pending.append(forecast.question_id)
                if self._corpus_writer is not None:
                    # The Stage-2 corpus row (question, workflow, evidence,
                    # forecast) is written now, while the conductor's workflow
                    # trace exists; the score job completes it with the
                    # resolution + proper score (§4 — routing is never hidden).
                    self._corpus_writer.capture(forecast.question_id, result.workflow)
            else:
                refused.append(question.question_id)
        return HarvestRun(pending=tuple(pending), refused=tuple(refused))
