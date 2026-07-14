"""Append-only, tamper-evident, event-sourced registry store.

The store is the immutable system of record. Public methods only ever *append*
records; there is no UPDATE or DELETE path at the application layer, and the
Postgres backend additionally forbids mutation at the database layer. Every
record is content-hashed and chained to the prior record in its stream, so any
retroactive edit is detectable via ``verify_chain``.

Two stream kinds exist:

* experiment streams (``stream_id == experiment_id``) hold the immutable
  Experiment input plus its append-only Result and Decision output events;
* strategy streams (``stream_id == strategy_id``) hold the Strategy genesis,
  its StrategyVersions, and the LifecycleEvents whose fold *is* the status.
"""

from __future__ import annotations

import json
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import structlog
from psycopg import sql

from core.registry.fingerprint import compute_record_hash, trial_fingerprint
from core.registry.models import (
    Decision,
    DecisionInput,
    EvidenceSet,
    EvidenceSetInput,
    Experiment,
    ExperimentInput,
    Forecast,
    ForecastInput,
    IllegalTransitionError,
    LifecycleEvent,
    LifecycleEventInput,
    LifecycleState,
    Question,
    QuestionInput,
    RecordKind,
    ReproMetadata,
    Resolution,
    ResolutionInput,
    Result,
    ResultInput,
    Strategy,
    StrategyInput,
    StrategyVersion,
    StrategyVersionInput,
    StreamKind,
    ensure_utc,
    fold_lifecycle,
    next_state,
)

__all__ = [
    "ChainVerification",
    "DuplicateTrialWarning",
    "IllegalTransitionError",
    "InMemoryRegistryStore",
    "IncompleteReproMetadataError",
    "PostgresRegistryStore",
    "RecordNotFoundError",
    "RegistryError",
    "RegistryEvent",
    "RegistryStore",
    "SecretInRecordError",
    "validate_repro_metadata",
]

_LOG = structlog.get_logger(__name__)
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Substrings that mark a payload key as likely holding a credential.
_SECRET_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "aws_secret",
)


class RegistryError(Exception):
    """Base class for registry errors."""


class RecordNotFoundError(RegistryError, KeyError):
    """Raised when a referenced record does not exist."""


class IncompleteReproMetadataError(RegistryError, ValueError):
    """Raised when an experiment's reproducibility bundle is incomplete."""


class SecretInRecordError(RegistryError, ValueError):
    """Raised when a record payload appears to contain a secret/credential."""


class DuplicateTrialWarning(UserWarning):
    """Marker type for a recorded experiment that duplicates an existing trial."""


@dataclass(frozen=True)
class RegistryEvent:
    """One immutable, chained record in the unified event log."""

    stream_id: str
    stream_kind: StreamKind
    seq: int
    record_kind: RecordKind
    record_id: str
    payload: dict[str, Any]
    prev_hash: str | None
    record_hash: str
    knowledge_time: datetime


@dataclass(frozen=True)
class ChainVerification:
    """Result of verifying a stream's hash chain."""

    stream_id: str
    ok: bool
    broken_at_seq: int | None = None
    broken_record_id: str | None = None
    reason: str | None = None


def validate_repro_metadata(meta: ReproMetadata) -> None:
    """Reject a reproducibility bundle missing any required field.

    Defense-in-depth precondition: the typed models already enforce presence,
    but this guards against bundles built via ``model_construct`` that bypass
    validation. Raises ``IncompleteReproMetadataError`` naming the first gap.
    """
    missing: list[str] = []
    if not (meta.code_sha and meta.code_sha.strip()):
        missing.append("code_sha")
    if not (meta.spec_hash and meta.spec_hash.strip()):
        missing.append("spec_hash")
    if not (meta.env and meta.env.python_version and meta.env.python_version.strip()):
        missing.append("env.python_version")
    if meta.data_snapshot is None:
        missing.append("data_snapshot")
    else:
        if meta.data_snapshot.as_of is None:
            missing.append("data_snapshot.as_of")
        if not meta.data_snapshot.universe_spec:
            missing.append("data_snapshot.universe_spec")
    if meta.params is None:
        missing.append("params")
    if meta.seeds is None:
        missing.append("seeds")
    if missing:
        msg = f"Incomplete reproducibility metadata; missing/blank: {', '.join(missing)}"
        raise IncompleteReproMetadataError(msg)


def _assert_no_secrets(payload: Any, *, _path: str = "payload") -> None:
    """Reject payloads whose keys look like they hold credentials.

    Contract (CLAUDE.md N3 / constraints): secrets never enter the record.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_l = str(key).lower()
            if any(marker in key_l for marker in _SECRET_MARKERS) and value:
                msg = (
                    f"Refusing to store a likely secret at {_path}.{key!r}; "
                    "records must reference versions/digests, never credentials."
                )
                raise SecretInRecordError(msg)
            _assert_no_secrets(value, _path=f"{_path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, value in enumerate(payload):
            _assert_no_secrets(value, _path=f"{_path}[{i}]")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class RegistryStore(ABC):
    """Append-only registry: public API is shared, storage primitives differ.

    Subclasses implement only ``_persist`` (atomic per-stream append) and the
    read primitives ``_stream_events`` / ``_events_of_kind``. All write
    semantics, hashing, chaining, fingerprinting, lifecycle folding, query
    logic, and ``verify_chain`` live here so both backends behave identically.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        return ensure_utc(self._clock())

    # --- storage primitives (subclass-provided) ------------------------------

    @abstractmethod
    def _persist(
        self,
        *,
        stream_id: str,
        stream_kind: StreamKind,
        record_kind: RecordKind,
        record_id: str,
        payload: dict[str, Any],
    ) -> RegistryEvent:
        """Atomically append one record to ``stream_id``.

        Must determine ``seq`` and ``prev_hash`` from the current stream tail
        under a per-stream lock, stamp ``knowledge_time`` from ``self._now()``,
        compute the record hash, and persist — never losing or reordering a
        concurrent append to the same stream.
        """

    @abstractmethod
    def _stream_events(self, stream_id: str) -> list[RegistryEvent]:
        """Return all events for a stream ordered by ascending ``seq``."""

    @abstractmethod
    def _events_of_kind(self, record_kind: RecordKind) -> list[RegistryEvent]:
        """Return all events of a given kind (cross-stream), order unspecified."""

    # --- write API -----------------------------------------------------------

    def record_experiment(self, exp: ExperimentInput) -> str:
        """Append an immutable, hashed, chained experiment record.

        Rejects any experiment whose reproducibility bundle is incomplete and
        any parent reference that does not resolve. Computes and stores the
        deterministic ``trial_fingerprint``; flags (does not block) a duplicate
        trial. The record is never mutated thereafter.
        """
        validate_repro_metadata(exp.repro)
        if exp.parent_experiment_id is not None:
            # Lineage integrity: a parent reference must resolve.
            self.get_experiment(exp.parent_experiment_id)

        experiment_id = _new_id("exp")
        fingerprint = trial_fingerprint(exp.repro)
        duplicates = self.duplicate_experiment_ids(fingerprint)

        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "trial_fingerprint": fingerprint,
            **exp.model_dump(mode="json"),
        }
        _assert_no_secrets(payload)
        self._persist(
            stream_id=experiment_id,
            stream_kind="experiment",
            record_kind="experiment",
            record_id=experiment_id,
            payload=payload,
        )
        if duplicates:
            _LOG.warning(
                "duplicate_trial_recorded",
                experiment_id=experiment_id,
                trial_fingerprint=fingerprint,
                prior_experiment_ids=list(duplicates),
            )
        return experiment_id

    def record_result(self, result: ResultInput) -> str:
        """Append a harness Result to its experiment's stream."""
        self.get_experiment(result.experiment_id)
        result_id = _new_id("res")
        payload = {"result_id": result_id, **result.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=result.experiment_id,
            stream_kind="experiment",
            record_kind="result",
            record_id=result_id,
            payload=payload,
        )
        return result_id

    def record_decision(self, decision: DecisionInput) -> str:
        """Append an attributable promote/reject/abandon Decision.

        A decision names the deciding component and version; it is chained into
        the experiment stream, not a forgeable "set status = promoted" write.
        """
        self.get_experiment(decision.experiment_id)
        decision_id = _new_id("dec")
        payload = {"decision_id": decision_id, **decision.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=decision.experiment_id,
            stream_kind="experiment",
            record_kind="decision",
            record_id=decision_id,
            payload=payload,
        )
        return decision_id

    def create_strategy(self, strategy: StrategyInput) -> str:
        """Open a strategy stream and emit its initial 'create' lifecycle event.

        The strategy starts in state ``candidate`` (the fold of ``[create]``).
        """
        strategy_id = _new_id("strat")
        payload = {"strategy_id": strategy_id, **strategy.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=strategy_id,
            stream_kind="strategy",
            record_kind="strategy",
            record_id=strategy_id,
            payload=payload,
        )
        self.record_lifecycle_event(
            LifecycleEventInput(
                strategy_id=strategy_id, event="create", rationale="strategy created"
            )
        )
        return strategy_id

    def record_strategy_version(self, version: StrategyVersionInput) -> str:
        """Append a StrategyVersion linked to its originating experiment."""
        self.get_strategy(version.strategy_id)
        self.get_experiment(version.origin_experiment_id)
        version_id = _new_id("ver")
        payload = {
            "strategy_version_id": version_id,
            **version.model_dump(mode="json"),
        }
        _assert_no_secrets(payload)
        self._persist(
            stream_id=version.strategy_id,
            stream_kind="strategy",
            record_kind="strategy_version",
            record_id=version_id,
            payload=payload,
        )
        return version_id

    def record_lifecycle_event(self, event: LifecycleEventInput) -> str:
        """Append a lifecycle transition after validating it against the fold.

        Illegal / out-of-order transitions (e.g. retire-before-promote) are
        rejected before anything is persisted.
        """
        self.get_strategy(event.strategy_id)
        current = self.current_state(event.strategy_id)
        next_state(current, event.event)  # raises IllegalTransitionError if illegal
        event_id = _new_id("lc")
        payload = {"lifecycle_event_id": event_id, **event.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=event.strategy_id,
            stream_kind="strategy",
            record_kind="lifecycle_event",
            record_id=event_id,
            payload=payload,
        )
        return event_id

    # --- forecast taxonomy: write API (DELPHI §3/§5) -------------------------

    def record_question(self, question: QuestionInput) -> str:
        """Open a question stream with its immutable genesis Question record.

        The question is the root of the forecast audit trail; EvidenceSets,
        Forecasts, and Resolutions are appended to this same stream.
        """
        question_id = _new_id("q")
        payload = {"question_id": question_id, **question.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=question_id,
            stream_kind="question",
            record_kind="question",
            record_id=question_id,
            payload=payload,
        )
        return question_id

    def record_evidence_set(self, evidence_set: EvidenceSetInput) -> str:
        """Append an as-of EvidenceSet to its question's stream."""
        self.get_question(evidence_set.question_id)
        evidence_set_id = _new_id("evs")
        payload = {"evidence_set_id": evidence_set_id, **evidence_set.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=evidence_set.question_id,
            stream_kind="question",
            record_kind="evidence_set",
            record_id=evidence_set_id,
            payload=payload,
        )
        return evidence_set_id

    def record_forecast(self, forecast: ForecastInput) -> str:
        """Append a formed Forecast to its question's stream.

        Any referenced ``evidence_set_id`` must resolve within the same stream.
        """
        self.get_question(forecast.question_id)
        if forecast.evidence_set_id is not None:
            self._require_evidence_set(forecast.question_id, forecast.evidence_set_id)
        forecast_id = _new_id("fc")
        payload = {"forecast_id": forecast_id, **forecast.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=forecast.question_id,
            stream_kind="question",
            record_kind="forecast",
            record_id=forecast_id,
            payload=payload,
        )
        return forecast_id

    def record_resolution(self, resolution: ResolutionInput) -> str:
        """Append a ground-truth Resolution to its question's stream.

        Any referenced ``forecast_id`` must resolve within the same stream.
        """
        self.get_question(resolution.question_id)
        if resolution.forecast_id is not None:
            self._require_forecast(resolution.question_id, resolution.forecast_id)
        resolution_id = _new_id("rsl")
        payload = {"resolution_id": resolution_id, **resolution.model_dump(mode="json")}
        _assert_no_secrets(payload)
        self._persist(
            stream_id=resolution.question_id,
            stream_kind="question",
            record_kind="resolution",
            record_id=resolution_id,
            payload=payload,
        )
        return resolution_id

    # --- forecast taxonomy: read API -----------------------------------------

    def get_question(self, question_id: str) -> Question:
        """Return the question genesis record, or raise ``RecordNotFoundError``."""
        events = self._stream_events(question_id)
        if not events or events[0].record_kind != "question":
            msg = f"No question with id {question_id!r}."
            raise RecordNotFoundError(msg)
        return _to_question(events[0])

    def evidence_sets_for(self, question_id: str) -> tuple[EvidenceSet, ...]:
        """All evidence sets for a question, in append order."""
        self.get_question(question_id)
        return tuple(
            _to_evidence_set(ev)
            for ev in self._stream_events(question_id)
            if ev.record_kind == "evidence_set"
        )

    def forecasts_for(self, question_id: str) -> tuple[Forecast, ...]:
        """All forecasts for a question, in append order."""
        self.get_question(question_id)
        return tuple(
            _to_forecast(ev)
            for ev in self._stream_events(question_id)
            if ev.record_kind == "forecast"
        )

    def resolutions_for(self, question_id: str) -> tuple[Resolution, ...]:
        """All resolutions for a question, in append order."""
        self.get_question(question_id)
        return tuple(
            _to_resolution(ev)
            for ev in self._stream_events(question_id)
            if ev.record_kind == "resolution"
        )

    def get_forecast(self, forecast_id: str) -> Forecast:
        """Return a forecast by its id (cross-stream), or raise ``RecordNotFoundError``."""
        for ev in self._events_of_kind("forecast"):
            if ev.record_id == forecast_id:
                return _to_forecast(ev)
        msg = f"No forecast with id {forecast_id!r}."
        raise RecordNotFoundError(msg)

    def resolutions_for_forecast(self, forecast_id: str) -> tuple[Resolution, ...]:
        """All resolutions that reference a specific forecast, in stream order."""
        return tuple(
            _to_resolution(ev)
            for ev in self._events_of_kind("resolution")
            if ev.payload.get("forecast_id") == forecast_id
        )

    def questions_by_domain(self, domain: str) -> tuple[Question, ...]:
        """All questions in a given domain, in knowledge-time order."""
        matched = [
            _to_question(ev)
            for ev in self._events_of_kind("question")
            if ev.payload.get("domain") == domain
        ]
        return tuple(sorted(matched, key=lambda q: (q.knowledge_time, q.question_id)))

    def all_questions(self) -> tuple[Question, ...]:
        """All recorded questions, in knowledge-time order (read-only enumeration)."""
        matched = [_to_question(ev) for ev in self._events_of_kind("question")]
        return tuple(sorted(matched, key=lambda q: (q.knowledge_time, q.question_id)))

    def _require_evidence_set(self, question_id: str, evidence_set_id: str) -> None:
        for ev in self._stream_events(question_id):
            if ev.record_kind == "evidence_set" and ev.record_id == evidence_set_id:
                return
        msg = f"No evidence set {evidence_set_id!r} in question {question_id!r}."
        raise RecordNotFoundError(msg)

    def _require_forecast(self, question_id: str, forecast_id: str) -> None:
        for ev in self._stream_events(question_id):
            if ev.record_kind == "forecast" and ev.record_id == forecast_id:
                return
        msg = f"No forecast {forecast_id!r} in question {question_id!r}."
        raise RecordNotFoundError(msg)

    # --- read API ------------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Return the experiment record, or raise ``RecordNotFoundError``."""
        events = self._stream_events(experiment_id)
        if not events or events[0].record_kind != "experiment":
            msg = f"No experiment with id {experiment_id!r}."
            raise RecordNotFoundError(msg)
        return _to_experiment(events[0])

    def results_for(self, experiment_id: str) -> tuple[Result, ...]:
        """All results for an experiment, in append order."""
        self.get_experiment(experiment_id)
        return tuple(
            _to_result(ev)
            for ev in self._stream_events(experiment_id)
            if ev.record_kind == "result"
        )

    def decisions_for(self, experiment_id: str) -> tuple[Decision, ...]:
        """All decisions for an experiment, in append order."""
        self.get_experiment(experiment_id)
        return tuple(
            _to_decision(ev)
            for ev in self._stream_events(experiment_id)
            if ev.record_kind == "decision"
        )

    def experiments_by_author(self, author: str) -> tuple[Experiment, ...]:
        """All experiments by a given author/agent."""
        return self._experiments_where(lambda ev: ev.payload.get("author") == author)

    def experiments_by_niche(self, niche: str) -> tuple[Experiment, ...]:
        """All experiments in a given niche (failures included)."""
        return self._experiments_where(lambda ev: ev.payload.get("niche") == niche)

    def all_experiments(self) -> tuple[Experiment, ...]:
        """All recorded experiments, in knowledge-time order.

        Contract: read-only enumeration for derived indexes (e.g. memory rebuild).
        Failures are included; outcome is resolved via ``decisions_for``.
        """
        return self._experiments_where(lambda _ev: True)

    def experiments_by_outcome(self, outcome: str) -> tuple[Experiment, ...]:
        """Experiments whose latest decision has ``outcome``.

        Failures (``reject`` / ``abandon``) are first-class and as queryable as
        successes.
        """
        latest: dict[str, RegistryEvent] = {}
        for ev in self._events_of_kind("decision"):
            eid = str(ev.payload["experiment_id"])
            current = latest.get(eid)
            if current is None or ev.seq > current.seq:
                latest[eid] = ev
        matched = [
            self.get_experiment(eid)
            for eid, ev in latest.items()
            if ev.payload.get("outcome") == outcome
        ]
        return tuple(sorted(matched, key=lambda e: (e.knowledge_time, e.experiment_id)))

    def duplicate_experiment_ids(self, fingerprint: str) -> tuple[str, ...]:
        """Experiment ids already recorded under ``fingerprint`` (dedup view)."""
        return tuple(
            str(ev.payload["experiment_id"])
            for ev in self._events_of_kind("experiment")
            if ev.payload.get("trial_fingerprint") == fingerprint
        )

    def experiment_children(self, experiment_id: str) -> tuple[Experiment, ...]:
        """Direct children of an experiment (lineage traversal, downward)."""
        self.get_experiment(experiment_id)
        children = [
            _to_experiment(ev)
            for ev in self._events_of_kind("experiment")
            if ev.payload.get("parent_experiment_id") == experiment_id
        ]
        return tuple(sorted(children, key=lambda e: (e.knowledge_time, e.experiment_id)))

    def experiment_lineage(self, experiment_id: str) -> tuple[Experiment, ...]:
        """Ancestry chain from root to ``experiment_id`` inclusive (upward)."""
        chain: list[Experiment] = []
        seen: set[str] = set()
        cursor: str | None = experiment_id
        while cursor is not None:
            if cursor in seen:
                msg = f"Cycle detected in experiment lineage at {cursor!r}."
                raise RegistryError(msg)
            seen.add(cursor)
            exp = self.get_experiment(cursor)
            chain.append(exp)
            cursor = exp.parent_experiment_id
        chain.reverse()
        return tuple(chain)

    def get_strategy(self, strategy_id: str) -> Strategy:
        """Return the strategy genesis record, or raise ``RecordNotFoundError``."""
        events = self._stream_events(strategy_id)
        if not events or events[0].record_kind != "strategy":
            msg = f"No strategy with id {strategy_id!r}."
            raise RecordNotFoundError(msg)
        return _to_strategy(events[0])

    def strategy_versions(self, strategy_id: str) -> tuple[StrategyVersion, ...]:
        """All versions of a strategy, in append order."""
        self.get_strategy(strategy_id)
        return tuple(
            _to_strategy_version(ev)
            for ev in self._stream_events(strategy_id)
            if ev.record_kind == "strategy_version"
        )

    def lifecycle_events(self, strategy_id: str) -> tuple[LifecycleEvent, ...]:
        """All lifecycle events of a strategy, in append order."""
        self.get_strategy(strategy_id)
        return tuple(
            _to_lifecycle_event(ev)
            for ev in self._stream_events(strategy_id)
            if ev.record_kind == "lifecycle_event"
        )

    def current_state(self, strategy_id: str) -> LifecycleState | None:
        """Current lifecycle state as the fold over LifecycleEvents."""
        self.get_strategy(strategy_id)
        events = [
            ev.payload["event"]
            for ev in self._stream_events(strategy_id)
            if ev.record_kind == "lifecycle_event"
        ]
        return fold_lifecycle(events)

    def strategies_by_state(self, state: LifecycleState) -> tuple[Strategy, ...]:
        """All strategies whose folded current state equals ``state``."""
        result: list[Strategy] = []
        for ev in self._events_of_kind("strategy"):
            strategy = _to_strategy(ev)
            if self.current_state(strategy.strategy_id) == state:
                result.append(strategy)
        return tuple(sorted(result, key=lambda s: (s.knowledge_time, s.strategy_id)))

    def strategy_version_ancestry(self, strategy_id: str, version: int) -> tuple[Experiment, ...]:
        """Full experiment ancestry behind a specific strategy version."""
        for ver in self.strategy_versions(strategy_id):
            if ver.version == version:
                return self.experiment_lineage(ver.origin_experiment_id)
        msg = f"Strategy {strategy_id!r} has no version {version}."
        raise RecordNotFoundError(msg)

    # --- tamper evidence -----------------------------------------------------

    def verify_chain(self, stream_id: str) -> ChainVerification:
        """Recompute hashes and confirm per-stream linkage.

        Returns ``ok=True`` for an intact (or empty) stream, otherwise the first
        broken link (seq + record id + reason).
        """
        events = self._stream_events(stream_id)
        prev_hash: str | None = None
        for index, ev in enumerate(events):
            if ev.seq != index:
                return ChainVerification(
                    stream_id=stream_id,
                    ok=False,
                    broken_at_seq=ev.seq,
                    broken_record_id=ev.record_id,
                    reason=f"non-contiguous seq: expected {index}, found {ev.seq}",
                )
            if ev.prev_hash != prev_hash:
                return ChainVerification(
                    stream_id=stream_id,
                    ok=False,
                    broken_at_seq=ev.seq,
                    broken_record_id=ev.record_id,
                    reason="prev_hash does not match prior record's hash",
                )
            recomputed = compute_record_hash(
                stream_id=ev.stream_id,
                seq=ev.seq,
                record_kind=ev.record_kind,
                record_id=ev.record_id,
                payload=ev.payload,
                prev_hash=ev.prev_hash,
                knowledge_time=ev.knowledge_time,
            )
            if recomputed != ev.record_hash:
                return ChainVerification(
                    stream_id=stream_id,
                    ok=False,
                    broken_at_seq=ev.seq,
                    broken_record_id=ev.record_id,
                    reason="record_hash does not match content (record was altered)",
                )
            prev_hash = ev.record_hash
        return ChainVerification(stream_id=stream_id, ok=True)

    # --- helpers -------------------------------------------------------------

    def _experiments_where(
        self, predicate: Callable[[RegistryEvent], bool]
    ) -> tuple[Experiment, ...]:
        matched = [_to_experiment(ev) for ev in self._events_of_kind("experiment") if predicate(ev)]
        return tuple(sorted(matched, key=lambda e: (e.knowledge_time, e.experiment_id)))

    def _build_event(
        self,
        *,
        stream_id: str,
        stream_kind: StreamKind,
        record_kind: RecordKind,
        record_id: str,
        payload: dict[str, Any],
        seq: int,
        prev_hash: str | None,
        knowledge_time: datetime,
    ) -> RegistryEvent:
        """Compute the chained hash and assemble an event (used inside ``_persist``)."""
        record_hash = compute_record_hash(
            stream_id=stream_id,
            seq=seq,
            record_kind=record_kind,
            record_id=record_id,
            payload=payload,
            prev_hash=prev_hash,
            knowledge_time=knowledge_time,
        )
        return RegistryEvent(
            stream_id=stream_id,
            stream_kind=stream_kind,
            seq=seq,
            record_kind=record_kind,
            record_id=record_id,
            payload=payload,
            prev_hash=prev_hash,
            record_hash=record_hash,
            knowledge_time=knowledge_time,
        )


# --- typed readers -----------------------------------------------------------


def _merge(ev: RegistryEvent) -> dict[str, Any]:
    return {**ev.payload, "knowledge_time": ev.knowledge_time}


def _to_experiment(ev: RegistryEvent) -> Experiment:
    return Experiment.model_validate(_merge(ev))


def _to_result(ev: RegistryEvent) -> Result:
    return Result.model_validate(_merge(ev))


def _to_decision(ev: RegistryEvent) -> Decision:
    return Decision.model_validate(_merge(ev))


def _to_strategy(ev: RegistryEvent) -> Strategy:
    return Strategy.model_validate(_merge(ev))


def _to_strategy_version(ev: RegistryEvent) -> StrategyVersion:
    return StrategyVersion.model_validate(_merge(ev))


def _to_lifecycle_event(ev: RegistryEvent) -> LifecycleEvent:
    return LifecycleEvent.model_validate({**_merge(ev), "seq": ev.seq})


def _to_question(ev: RegistryEvent) -> Question:
    return Question.model_validate(_merge(ev))


def _to_evidence_set(ev: RegistryEvent) -> EvidenceSet:
    return EvidenceSet.model_validate(_merge(ev))


def _to_forecast(ev: RegistryEvent) -> Forecast:
    return Forecast.model_validate(_merge(ev))


def _to_resolution(ev: RegistryEvent) -> Resolution:
    return Resolution.model_validate(_merge(ev))


# --- in-memory backend -------------------------------------------------------


class InMemoryRegistryStore(RegistryStore):
    """Deterministic reference backend backed by Python collections.

    Uses per-stream locks so concurrent appends to *different* streams never
    block each other while same-stream appends serialize to keep the chain valid.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        super().__init__(clock=clock)
        self._events: list[RegistryEvent] = []
        self._by_stream: dict[str, list[RegistryEvent]] = {}
        self._meta_lock = threading.Lock()
        self._stream_locks: dict[str, threading.Lock] = {}

    def _stream_lock(self, stream_id: str) -> threading.Lock:
        with self._meta_lock:
            return self._stream_locks.setdefault(stream_id, threading.Lock())

    def _persist(
        self,
        *,
        stream_id: str,
        stream_kind: StreamKind,
        record_kind: RecordKind,
        record_id: str,
        payload: dict[str, Any],
    ) -> RegistryEvent:
        with self._stream_lock(stream_id):
            with self._meta_lock:
                tail = self._by_stream.get(stream_id, [])
                seq = len(tail)
                prev_hash = tail[-1].record_hash if tail else None
            event = self._build_event(
                stream_id=stream_id,
                stream_kind=stream_kind,
                record_kind=record_kind,
                record_id=record_id,
                payload=payload,
                seq=seq,
                prev_hash=prev_hash,
                knowledge_time=self._now(),
            )
            with self._meta_lock:
                self._events.append(event)
                self._by_stream.setdefault(stream_id, []).append(event)
            return event

    def _stream_events(self, stream_id: str) -> list[RegistryEvent]:
        with self._meta_lock:
            return list(self._by_stream.get(stream_id, []))

    def _events_of_kind(self, record_kind: RecordKind) -> list[RegistryEvent]:
        with self._meta_lock:
            return [ev for ev in self._events if ev.record_kind == record_kind]


# --- postgres backend --------------------------------------------------------


class PostgresRegistryStore(RegistryStore):
    """PostgreSQL-backed registry. Appends are atomic per stream via advisory
    locks; UPDATE/DELETE are forbidden by database triggers."""

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._conn = conn

    @classmethod
    def connect(
        cls,
        dsn: str,
        *,
        migrate: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> PostgresRegistryStore:
        """Open a connection and optionally apply pending migrations.

        The connection uses ``autocommit=True`` so each append commits atomically
        via its own ``conn.transaction()`` block. With autocommit *off*, a read
        issued before the write (e.g. ``get_question`` inside ``record_forecast``)
        opens an implicit transaction, which demotes the write's ``transaction()``
        to a mere savepoint that is never committed — silently dropping the write
        on connection close (a §3 "no silent forecasts" violation).
        """
        conn = psycopg.connect(dsn, autocommit=True)
        store = cls(conn, clock=clock)
        if migrate:
            store.apply_migrations()
        return store

    def apply_migrations(self) -> None:
        """Apply SQL migrations from ``registry/migrations/`` in sorted order."""
        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        # One transaction for all migrations (atomic under autocommit=True).
        with self._conn.transaction(), self._conn.cursor() as cur:
            for path in migration_files:
                cur.execute(sql.SQL(path.read_text()))  # pyright: ignore[reportArgumentType]

    def _persist(
        self,
        *,
        stream_id: str,
        stream_kind: StreamKind,
        record_kind: RecordKind,
        record_id: str,
        payload: dict[str, Any],
    ) -> RegistryEvent:
        with self._conn.transaction(), self._conn.cursor() as cur:
            # Serialize appends to this stream only (other streams proceed).
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (stream_id,))
            cur.execute(
                """
                    SELECT seq, record_hash FROM registry_events
                    WHERE stream_id = %s ORDER BY seq DESC LIMIT 1
                    """,
                (stream_id,),
            )
            row = cur.fetchone()
            seq = row[0] + 1 if row else 0
            prev_hash = row[1] if row else None
            event = self._build_event(
                stream_id=stream_id,
                stream_kind=stream_kind,
                record_kind=record_kind,
                record_id=record_id,
                payload=payload,
                seq=seq,
                prev_hash=prev_hash,
                knowledge_time=self._now(),
            )
            cur.execute(
                """
                    INSERT INTO registry_events (
                        stream_id, stream_kind, seq, record_kind, record_id,
                        payload, prev_hash, record_hash, knowledge_time
                    ) VALUES (
                        %(stream_id)s, %(stream_kind)s, %(seq)s, %(record_kind)s,
                        %(record_id)s, %(payload)s, %(prev_hash)s, %(record_hash)s,
                        %(knowledge_time)s
                    )
                    """,
                {
                    "stream_id": event.stream_id,
                    "stream_kind": event.stream_kind,
                    "seq": event.seq,
                    "record_kind": event.record_kind,
                    "record_id": event.record_id,
                    "payload": json.dumps(event.payload),
                    "prev_hash": event.prev_hash,
                    "record_hash": event.record_hash,
                    "knowledge_time": event.knowledge_time,
                },
            )
        return event

    def _row_to_event(self, row: Sequence[Any]) -> RegistryEvent:
        return RegistryEvent(
            stream_id=row[0],
            stream_kind=row[1],
            seq=row[2],
            record_kind=row[3],
            record_id=row[4],
            payload=_parse_jsonb(row[5]),
            prev_hash=row[6],
            record_hash=row[7],
            knowledge_time=ensure_utc(row[8]),
        )

    def _fetch(self, where: str, params: dict[str, Any], order: str) -> list[RegistryEvent]:
        query = (
            "SELECT stream_id, stream_kind, seq, record_kind, record_id, "
            "payload, prev_hash, record_hash, knowledge_time "
            f"FROM registry_events WHERE {where} ORDER BY {order}"
        )
        with self._conn.cursor() as cur:
            cur.execute(query, params)  # pyright: ignore[reportArgumentType]
            rows = cur.fetchall()
        return [self._row_to_event(row) for row in rows]

    def _stream_events(self, stream_id: str) -> list[RegistryEvent]:
        return self._fetch("stream_id = %(stream_id)s", {"stream_id": stream_id}, "seq")

    def _events_of_kind(self, record_kind: RecordKind) -> list[RegistryEvent]:
        return self._fetch("record_kind = %(kind)s", {"kind": record_kind}, "stream_id, seq")

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> PostgresRegistryStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _parse_jsonb(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        loaded: Any = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    msg = f"Unexpected JSONB payload type: {type(raw)!r}"
    raise TypeError(msg)
