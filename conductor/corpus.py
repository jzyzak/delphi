"""Corpus tuple writer (C8.3) — the Stage-2 training data.

Assembles the scored ``(question, workflow, evidence, forecast, resolution,
proper-score)`` tuple from the immutable registry records and the conductor's
workflow trace, and writes it to the corpus store. This is the asset that
compounds with every resolved question (CLAUDE.md §1/§4): you cannot train a
learned conductor until this corpus exists, and every heuristic run adds one row.
The proper score is filled in only once a resolution exists (never accuracy, §2.3).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from conductor.heuristic import WorkflowTrace
from core.registry.models import EvidenceSet, Forecast, Question, Resolution
from core.registry.store import RegistryStore
from evaluation.scoring import BrierScorer, Scorer

__all__ = [
    "CorpusStore",
    "CorpusTuple",
    "CorpusWriter",
    "InMemoryCorpusStore",
]


@dataclass(frozen=True)
class CorpusTuple:
    """A scored training tuple: question -> workflow -> evidence -> forecast -> outcome."""

    question: Question
    workflow: dict[str, Any]
    evidence: tuple[EvidenceSet, ...]
    forecast: Forecast
    resolution: Resolution | None
    proper_score: float | None
    scorer: str

    @property
    def resolved(self) -> bool:
        return self.resolution is not None


@runtime_checkable
class CorpusStore(Protocol):
    """Append-only sink for scored corpus tuples."""

    def write(self, tuple_: CorpusTuple) -> None:
        """Persist one corpus tuple."""
        ...


class InMemoryCorpusStore:
    """In-memory corpus store for tests and single-process runs."""

    def __init__(self) -> None:
        self._tuples: list[CorpusTuple] = []

    def write(self, tuple_: CorpusTuple) -> None:
        self._tuples.append(tuple_)

    def __len__(self) -> int:
        return len(self._tuples)

    def __iter__(self) -> Iterator[CorpusTuple]:
        return iter(self._tuples)


class CorpusWriter:
    """Assembles scored corpus tuples from the registry + a workflow trace."""

    def __init__(
        self,
        *,
        store: RegistryStore,
        corpus: CorpusStore,
        scorer: Scorer | None = None,
    ) -> None:
        self._store = store
        self._corpus = corpus
        self._scorer = scorer if scorer is not None else BrierScorer()

    def capture(self, question_id: str, workflow: WorkflowTrace) -> CorpusTuple:
        """Assemble + write the scored tuple for ``question_id`` (latest forecast)."""
        question = self._store.get_question(question_id)
        forecasts = self._store.forecasts_for(question_id)
        if not forecasts:
            msg = f"no forecast recorded for question {question_id}; nothing to capture."
            raise ValueError(msg)
        forecast = forecasts[-1]
        evidence = self._store.evidence_sets_for(question_id)
        resolutions = self._store.resolutions_for(question_id)
        resolution = resolutions[-1] if resolutions else None

        proper_score = self._score(forecast, resolution)
        corpus_tuple = CorpusTuple(
            question=question,
            workflow=workflow.as_dict(),
            evidence=evidence,
            forecast=forecast,
            resolution=resolution,
            proper_score=proper_score,
            scorer=self._scorer.name,
        )
        self._corpus.write(corpus_tuple)
        return corpus_tuple

    def _score(self, forecast: Forecast, resolution: Resolution | None) -> float | None:
        if resolution is None or forecast.probability is None:
            return None
        if resolution.resolved_value not in (0.0, 1.0):
            return None  # non-binary outcomes are scored via CRPS elsewhere
        return self._scorer.score(forecast.probability, resolution.resolved_value)
