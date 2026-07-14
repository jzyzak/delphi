"""Derived vector index over registry experiments.

Contract: memory is a rebuildable cache. The registry remains canonical; this
module never mutates registry records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psycopg
import structlog
from pgvector.psycopg import register_vector
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field

from core.memory.embedder import Embedder
from core.registry.fingerprint import canonical_json
from core.registry.models import Decision, Experiment, ReproMetadata, Result
from core.registry.store import RegistryStore

_LOG = structlog.get_logger(__name__)
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

RecallOutcome = Literal["promoted", "rejected", "abandoned", "any"]
IndexOutcome = Literal["promoted", "rejected", "abandoned", "pending"]

_OUTCOME_FROM_DECISION: dict[str, IndexOutcome] = {
    "promote": "promoted",
    "reject": "rejected",
    "abandon": "abandoned",
}


class MemoryError(Exception):
    """Base class for agent memory errors."""


class DimensionMismatchError(MemoryError, ValueError):
    """Raised when an embedding dimension does not match the index."""


class Recollection(BaseModel):
    """One recalled experiment with similarity score and derived lessons."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    niche: str
    outcome: IndexOutcome
    score: float = Field(ge=-1.0, le=1.0)
    embedded_text: str
    lessons: str
    trial_fingerprint: str


@dataclass(frozen=True)
class IndexDocument:
    """Assembled index payload for one experiment."""

    experiment_id: str
    niche: str
    outcome: IndexOutcome
    trial_fingerprint: str
    embedded_text: str
    lessons: str
    knowledge_time: datetime


def assemble_document(store: RegistryStore, experiment: Experiment) -> IndexDocument:
    """Fold registry records into embeddable text and filter fields.

    Contract: read-only over ``store``. Outcome comes from the latest decision;
    lessons are derived from decision rationale/evidence and result metrics.
    """
    decisions = store.decisions_for(experiment.experiment_id)
    results = store.results_for(experiment.experiment_id)
    latest_decision = decisions[-1] if decisions else None
    latest_result = results[-1] if results else None

    outcome: IndexOutcome = (
        _OUTCOME_FROM_DECISION[latest_decision.outcome]
        if latest_decision is not None
        else "pending"
    )
    lessons = _derive_lessons(latest_decision, latest_result)
    spec_description = render_spec_description(experiment.repro)
    embedded_text = _build_embedded_text(
        hypothesis=experiment.hypothesis,
        economic_rationale=experiment.economic_rationale,
        spec_description=spec_description,
        outcome=outcome,
        lessons=lessons,
    )

    return IndexDocument(
        experiment_id=experiment.experiment_id,
        niche=experiment.niche,
        outcome=outcome,
        trial_fingerprint=experiment.trial_fingerprint,
        embedded_text=embedded_text,
        lessons=lessons,
        knowledge_time=experiment.knowledge_time,
    )


def render_spec_description(repro: ReproMetadata) -> str:
    """Render a deterministic spec description from registry repro metadata."""
    params_json = canonical_json(repro.params)
    return f"{repro.spec_kind} spec (hash={repro.spec_hash}), params={params_json}"


def _derive_lessons(decision: Decision | None, result: Result | None) -> str:
    parts: list[str] = []
    if decision is not None:
        if decision.rationale.strip():
            parts.append(decision.rationale.strip())
        if decision.evidence:
            parts.append(f"evidence={canonical_json(decision.evidence)}")
    if result is not None and result.metrics:
        parts.append(f"metrics={canonical_json(result.metrics)}")
    return " | ".join(parts)


def _build_embedded_text(
    *,
    hypothesis: str,
    economic_rationale: str,
    spec_description: str,
    outcome: IndexOutcome,
    lessons: str,
) -> str:
    lines = [
        f"Hypothesis: {hypothesis}",
        f"Economic rationale: {economic_rationale}",
        f"Spec: {spec_description}",
        f"Outcome: {outcome}",
    ]
    if lessons:
        lines.append(f"Lessons: {lessons}")
    return "\n".join(lines)


def _cosine_similarity(query: Sequence[float], candidate: Sequence[float]) -> float:
    left = np.asarray(query, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        msg = f"Vector shape mismatch: {left.shape} vs {right.shape}."
        raise DimensionMismatchError(msg)
    norm_left = float(np.linalg.norm(left))
    norm_right = float(np.linalg.norm(right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return float(np.dot(left, right) / (norm_left * norm_right))


def _validate_recall_outcome(outcome: RecallOutcome) -> None:
    if outcome not in ("promoted", "rejected", "abandoned", "any"):
        msg = f"Invalid recall outcome filter: {outcome!r}."
        raise ValueError(msg)


def _matches_outcome(index_outcome: IndexOutcome, filter_outcome: RecallOutcome) -> bool:
    if filter_outcome == "any":
        return True
    return index_outcome == filter_outcome


class VectorIndex(ABC):
    """Rebuildable embedding index over registry experiments."""

    def __init__(self, store: RegistryStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def index(self, experiment: Experiment) -> None:
        """Add or update the embedding for one registry experiment."""
        doc = assemble_document(self._store, experiment)
        vector = self._embed_one(doc.embedded_text)
        self._upsert(doc, vector)
        _LOG.info(
            "memory_indexed",
            experiment_id=experiment.experiment_id,
            niche=experiment.niche,
            outcome=doc.outcome,
        )

    def rebuild_from_registry(self) -> int:
        """Clear and reconstruct the entire index from the canonical registry."""
        self._clear()
        experiments = self._store.all_experiments()
        for experiment in experiments:
            self.index(experiment)
        _LOG.info("memory_rebuild", count=len(experiments))
        return len(experiments)

    def search(
        self,
        vector: Sequence[float],
        *,
        niche: str | None = None,
        outcome: RecallOutcome = "any",
        k: int = 10,
    ) -> list[Recollection]:
        """Return up to ``k`` nearest neighbors with cosine similarity scores."""
        _validate_recall_outcome(outcome)
        if k < 0:
            msg = "k must be non-negative."
            raise ValueError(msg)
        return self._search_vectors(vector, niche=niche, outcome=outcome, k=k)

    def _embed_one(self, text: str) -> list[float]:
        vectors = self._embedder.embed([text])
        if not vectors:
            msg = "Embedder returned no vectors."
            raise MemoryError(msg)
        vector = vectors[0]
        if len(vector) != self._embedder.dim:
            msg = f"Embedder returned dimension {len(vector)}; expected {self._embedder.dim}."
            raise DimensionMismatchError(msg)
        return vector

    @abstractmethod
    def _upsert(self, doc: IndexDocument, vector: Sequence[float]) -> None: ...

    @abstractmethod
    def _clear(self) -> None: ...

    @abstractmethod
    def _search_vectors(
        self,
        vector: Sequence[float],
        *,
        niche: str | None,
        outcome: RecallOutcome,
        k: int,
    ) -> list[Recollection]: ...


class InMemoryVectorIndex(VectorIndex):
    """In-process vector index for offline tests and local development."""

    def __init__(self, store: RegistryStore, embedder: Embedder) -> None:
        super().__init__(store, embedder)
        self._entries: dict[str, tuple[IndexDocument, list[float]]] = {}

    def _upsert(self, doc: IndexDocument, vector: Sequence[float]) -> None:
        self._entries[doc.experiment_id] = (doc, list(vector))

    def _clear(self) -> None:
        self._entries.clear()

    def _search_vectors(
        self,
        vector: Sequence[float],
        *,
        niche: str | None,
        outcome: RecallOutcome,
        k: int,
    ) -> list[Recollection]:
        scored: list[Recollection] = []
        for doc, stored_vector in self._entries.values():
            if niche is not None and doc.niche != niche:
                continue
            if not _matches_outcome(doc.outcome, outcome):
                continue
            score = _cosine_similarity(vector, stored_vector)
            scored.append(
                Recollection(
                    experiment_id=doc.experiment_id,
                    niche=doc.niche,
                    outcome=doc.outcome,
                    score=score,
                    embedded_text=doc.embedded_text,
                    lessons=doc.lessons,
                    trial_fingerprint=doc.trial_fingerprint,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.experiment_id))
        return scored[:k]


class PostgresVectorIndex(VectorIndex):
    """PostgreSQL + pgvector index. Fully rebuildable from the registry."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        store: RegistryStore,
        embedder: Embedder,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(store, embedder)
        self._conn = conn
        self._clock = clock or (lambda: datetime.now(UTC))
        register_vector(self._conn)

    @classmethod
    def connect(
        cls,
        dsn: str,
        store: RegistryStore,
        embedder: Embedder,
        *,
        migrate: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> PostgresVectorIndex:
        """Open a connection and optionally apply pending migrations."""
        conn = psycopg.connect(dsn)
        index = cls(conn, store, embedder, clock=clock)
        if migrate:
            index.apply_migrations()
        return index

    def apply_migrations(self) -> None:
        """Apply SQL migrations from ``research/memory/migrations/`` in sorted order.

        The ``{embedding_dim}`` placeholder in the schema is rendered from the
        embedder's ``dim`` so the pgvector column always matches the vectors it
        will store (single source of truth). Dimension is fixed at first apply;
        changing it later requires a fresh index rebuild.
        """
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        with self._conn.cursor() as cur:
            for path in migration_files:
                rendered = path.read_text().replace("{embedding_dim}", str(self._embedder.dim))
                cur.execute(sql.SQL(rendered))  # pyright: ignore[reportArgumentType]
        self._conn.commit()

    def _upsert(self, doc: IndexDocument, vector: Sequence[float]) -> None:
        now = self._clock()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_index (
                    experiment_id, niche, outcome, trial_fingerprint,
                    embedded_text, lessons, embedding, knowledge_time, updated_at
                )
                VALUES (
                    %(experiment_id)s, %(niche)s, %(outcome)s, %(trial_fingerprint)s,
                    %(embedded_text)s, %(lessons)s, %(embedding)s,
                    %(knowledge_time)s, %(updated_at)s
                )
                ON CONFLICT (experiment_id) DO UPDATE SET
                    niche = EXCLUDED.niche,
                    outcome = EXCLUDED.outcome,
                    trial_fingerprint = EXCLUDED.trial_fingerprint,
                    embedded_text = EXCLUDED.embedded_text,
                    lessons = EXCLUDED.lessons,
                    embedding = EXCLUDED.embedding,
                    knowledge_time = EXCLUDED.knowledge_time,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "experiment_id": doc.experiment_id,
                    "niche": doc.niche,
                    "outcome": doc.outcome,
                    "trial_fingerprint": doc.trial_fingerprint,
                    "embedded_text": doc.embedded_text,
                    "lessons": doc.lessons,
                    "embedding": list(vector),
                    "knowledge_time": doc.knowledge_time,
                    "updated_at": now,
                },
            )
        self._conn.commit()

    def _clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("TRUNCATE memory_index")
        self._conn.commit()

    def _search_vectors(
        self,
        vector: Sequence[float],
        *,
        niche: str | None,
        outcome: RecallOutcome,
        k: int,
    ) -> list[Recollection]:
        clauses = ["TRUE"]
        params: dict[str, Any] = {
            "embedding": list(vector),
            "k": k,
        }
        if niche is not None:
            clauses.append("niche = %(niche)s")
            params["niche"] = niche
        if outcome != "any":
            clauses.append("outcome = %(outcome)s")
            params["outcome"] = outcome

        where_sql = " AND ".join(clauses)
        query = f"""
            SELECT
                experiment_id,
                niche,
                outcome,
                trial_fingerprint,
                embedded_text,
                lessons,
                1 - (embedding <=> %(embedding)s::vector) AS score
            FROM memory_index
            WHERE {where_sql}
            ORDER BY embedding <=> %(embedding)s::vector
            LIMIT %(k)s
        """
        with self._conn.cursor() as cur:
            cur.execute(query, params)  # pyright: ignore[reportArgumentType]
            rows = cur.fetchall()

        return [
            Recollection(
                experiment_id=row[0],
                niche=row[1],
                outcome=row[2],
                trial_fingerprint=row[3],
                embedded_text=row[4],
                lessons=row[5],
                score=float(row[6]),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> PostgresVectorIndex:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def index_experiment(
    store: RegistryStore,
    index: VectorIndex,
    experiment_id: str,
) -> None:
    """Index one experiment by id; raises ``RecordNotFoundError`` if missing."""
    experiment = store.get_experiment(experiment_id)
    index.index(experiment)
