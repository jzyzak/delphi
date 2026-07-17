"""Corpus tuple writer (C8.3) — the Stage-2 training data.

Assembles the scored ``(question, workflow, evidence, forecast, resolution,
proper-score)`` tuple from the immutable registry records and the conductor's
workflow trace, and writes it to the corpus store. This is the asset that
compounds with every resolved question (CLAUDE.md §1/§4): you cannot train a
learned conductor until this corpus exists, and every heuristic run adds one row.
The proper score is filled in only once a resolution exists (never accuracy, §2.3).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from conductor.heuristic import WorkflowTrace
from core.registry.models import EvidenceSet, Forecast, Question, Resolution
from core.registry.store import RegistryStore
from evaluation.scoring import BrierScorer, Scorer

__all__ = [
    "CorpusStore",
    "CorpusTuple",
    "CorpusWriter",
    "FileCorpusStore",
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
    """Append-only sink for scored corpus tuples, with latest-row read-back."""

    def write(self, tuple_: CorpusTuple) -> None:
        """Persist one corpus tuple."""
        ...

    def latest_for(self, question_id: str) -> CorpusTuple | None:
        """The most recently written tuple for ``question_id`` (or ``None``)."""
        ...


class InMemoryCorpusStore:
    """In-memory corpus store for tests and single-process runs."""

    def __init__(self) -> None:
        self._tuples: list[CorpusTuple] = []

    def write(self, tuple_: CorpusTuple) -> None:
        self._tuples.append(tuple_)

    def latest_for(self, question_id: str) -> CorpusTuple | None:
        for tuple_ in reversed(self._tuples):
            if tuple_.question.question_id == question_id:
                return tuple_
        return None

    def __len__(self) -> int:
        return len(self._tuples)

    def __iter__(self) -> Iterator[CorpusTuple]:
        return iter(self._tuples)


def _tuple_to_payload(tuple_: CorpusTuple) -> dict[str, Any]:
    return {
        "question": tuple_.question.model_dump(mode="json"),
        "workflow": tuple_.workflow,
        "evidence": [e.model_dump(mode="json") for e in tuple_.evidence],
        "forecast": tuple_.forecast.model_dump(mode="json"),
        "resolution": (
            tuple_.resolution.model_dump(mode="json") if tuple_.resolution is not None else None
        ),
        "proper_score": tuple_.proper_score,
        "scorer": tuple_.scorer,
    }


def _tuple_from_payload(payload: dict[str, Any]) -> CorpusTuple:
    resolution = payload.get("resolution")
    return CorpusTuple(
        question=Question.model_validate(payload["question"]),
        workflow=dict(payload["workflow"]),
        evidence=tuple(EvidenceSet.model_validate(e) for e in payload["evidence"]),
        forecast=Forecast.model_validate(payload["forecast"]),
        resolution=Resolution.model_validate(resolution) if resolution is not None else None,
        proper_score=payload.get("proper_score"),
        scorer=str(payload["scorer"]),
    )


class FileCorpusStore:
    """Append-only JSONL corpus store — the durable single-host sink.

    One tuple per line; an existing file is loaded on construction so
    ``latest_for`` sees rows from prior processes (the harvest and score jobs
    run as separate ticks). A corrupt line fails loudly — a silently truncated
    training corpus is worse than a crashed tick.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._tuples: list[CorpusTuple] = []
        if self._path.exists():
            for i, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    self._tuples.append(_tuple_from_payload(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    msg = f"corpus store {self._path} is corrupt at line {i}."
                    raise ValueError(msg) from exc

    def write(self, tuple_: CorpusTuple) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_tuple_to_payload(tuple_), sort_keys=True, default=str) + "\n")
            fh.flush()
        self._tuples.append(tuple_)

    def latest_for(self, question_id: str) -> CorpusTuple | None:
        for tuple_ in reversed(self._tuples):
            if tuple_.question.question_id == question_id:
                return tuple_
        return None

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
        return self._assemble(question_id, workflow.as_dict())

    def refresh(self, question_id: str) -> CorpusTuple | None:
        """Complete a pending tuple once its question has resolved.

        The workflow trace only exists in memory at harvest time, so the score
        job replays it from the stored pending row. Returns the already-scored
        tuple unchanged, ``None`` when there is nothing to do (no pending row,
        or still unresolved), and otherwise writes + returns the scored tuple.
        """
        pending = self._corpus.latest_for(question_id)
        if pending is None:
            return None
        if pending.resolved:
            return pending
        if not self._store.resolutions_for(question_id):
            return None
        return self._assemble(question_id, dict(pending.workflow))

    def _assemble(self, question_id: str, workflow: dict[str, Any]) -> CorpusTuple:
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
            workflow=workflow,
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
