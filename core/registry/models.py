"""Typed domain models for the experiment & strategy registry.

The registry is append-only and event-sourced. These models describe:

* the reproducibility contract (``DataSnapshot`` / ``EnvFingerprint`` /
  ``ReproMetadata``) that every experiment write must satisfy,
* the input bundles callers submit (``*Input`` models),
* the stored records returned by the query API (``Experiment`` / ``Result`` /
  ``Decision`` / ``Strategy`` / ``StrategyVersion`` / ``LifecycleEvent``), and
* the strategy lifecycle state machine (a fold over append-only events).

All timestamps are tz-aware UTC; naive datetimes are rejected at the boundary.
Nothing here mutates persisted state — corrections are new records, never edits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Stream / record taxonomy ------------------------------------------------

StreamKind = Literal["experiment", "strategy", "question"]
RecordKind = Literal[
    "experiment",
    "result",
    "decision",
    "strategy",
    "strategy_version",
    "lifecycle_event",
    # Forecast taxonomy (DELPHI §3/§5): one question stream holds its genesis
    # Question, the EvidenceSet(s) gathered as-of, the Forecast(s) formed, and
    # the eventual Resolution(s). This is the immutable audit trail + corpus.
    "question",
    "evidence_set",
    "forecast",
    "resolution",
]

# Forecast question shapes DELPHI can resolve (CLAUDE.md §1/§3).
QuestionType = Literal["binary", "numeric", "multiple_choice", "date"]

# An experiment's terminal disposition. Failures (reject/abandon) are first-class.
DecisionOutcome = Literal["promote", "reject", "abandon"]

# Harness result status. ``error`` denotes a run that did not complete cleanly.
ResultStatus = Literal["success", "failure", "error"]

SpecKind = Literal["dsl", "code"]


def ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes and normalize to UTC.

    Contract: every timestamp entering or leaving the registry is tz-aware UTC.
    """
    if dt.tzinfo is None:
        msg = "Naive datetimes are not allowed; provide a tz-aware UTC datetime."
        raise ValueError(msg)
    return dt.astimezone(UTC)


def _require_nonempty(value: str, field: str) -> str:
    if not value or not value.strip():
        msg = f"Reproducibility metadata field {field!r} must be a non-empty string."
        raise ValueError(msg)
    return value


# --- Reproducibility contract ------------------------------------------------


class DataSnapshot(BaseModel):
    """PIT-native description of the dataset an experiment ran against.

    Contract: ``as_of`` is the knowledge-time ceiling; ``universe_spec`` is the
    classification + listing filter (from prompt 02). Together they make the
    exact dataset reconstructible from the bitemporal store — no raw rows copied.
    """

    model_config = ConfigDict(frozen=True)

    as_of: datetime
    universe_spec: dict[str, Any]

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _universe_spec_present(self) -> Self:
        if not self.universe_spec:
            msg = "DataSnapshot.universe_spec must be a non-empty mapping."
            raise ValueError(msg)
        return self


class EnvFingerprint(BaseModel):
    """Environment fingerprint: versions/digests only, never credentials.

    Contract (CLAUDE.md N3): references *what* ran (python + key package
    versions, container image digest), and contains no secrets.
    """

    model_config = ConfigDict(frozen=True)

    python_version: str
    packages: dict[str, str] = {}
    image_digest: str | None = None

    @field_validator("python_version")
    @classmethod
    def _python_version_present(cls, v: str) -> str:
        return _require_nonempty(v, "env.python_version")


class ReproMetadata(BaseModel):
    """Complete reproducibility bundle required to record an experiment.

    Contract: an experiment that cannot be reproduced is not recorded. All
    fields below are required; the critical string fields must be non-empty.
    """

    model_config = ConfigDict(frozen=True)

    code_sha: str
    dirty: bool
    spec_kind: SpecKind
    spec_hash: str
    params: dict[str, Any]
    data_snapshot: DataSnapshot
    env: EnvFingerprint
    seeds: dict[str, int]

    @field_validator("code_sha")
    @classmethod
    def _code_sha_present(cls, v: str) -> str:
        return _require_nonempty(v, "code_sha")

    @field_validator("spec_hash")
    @classmethod
    def _spec_hash_present(cls, v: str) -> str:
        return _require_nonempty(v, "spec_hash")


# --- Input bundles (what callers submit) -------------------------------------


class ExperimentInput(BaseModel):
    """Immutable input bundle for a single experiment.

    Contract: carries the falsifiable economic mechanism (CLAUDE.md §10) and a
    complete ``ReproMetadata`` bundle. ``parent_experiment_id`` records lineage.
    """

    model_config = ConfigDict(frozen=True)

    hypothesis: str
    economic_rationale: str
    author: str
    niche: str
    repro: ReproMetadata
    parent_experiment_id: str | None = None

    @field_validator("hypothesis", "economic_rationale", "author", "niche")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "Experiment hypothesis, rationale, author, and niche are required."
            raise ValueError(msg)
        return v


class ResultInput(BaseModel):
    """Append-only harness output referencing an experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    status: ResultStatus
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}


class DecisionInput(BaseModel):
    """Append-only promote/reject/abandon decision referencing an experiment.

    Contract: a decision names the deciding component and its version and is not
    a generic "set status = promoted" write — it is attributable and chained.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    outcome: DecisionOutcome
    deciding_component: str
    component_version: str
    rationale: str
    evidence: dict[str, Any] = {}

    @field_validator("deciding_component", "component_version", "rationale")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "Decision requires deciding_component, component_version, and rationale."
            raise ValueError(msg)
        return v


class StrategyInput(BaseModel):
    """Genesis record for a strategy stream."""

    model_config = ConfigDict(frozen=True)

    name: str
    niche: str
    author: str

    @field_validator("name", "niche", "author")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "Strategy name, niche, and author are required."
            raise ValueError(msg)
        return v


class StrategyVersionInput(BaseModel):
    """A concrete, versioned instantiation of a strategy.

    Contract: ``origin_experiment_id`` links the version to the experiment that
    produced it, anchoring full ancestry traversal (lineage).
    """

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    version: int
    origin_experiment_id: str
    spec_hash: str
    params: dict[str, Any] = {}

    @field_validator("version")
    @classmethod
    def _version_positive(cls, v: int) -> int:
        if v < 1:
            msg = "StrategyVersion.version must be >= 1."
            raise ValueError(msg)
        return v


LifecycleEventType = Literal["create", "promote", "retire"]
LifecycleState = Literal["candidate", "promoted", "retired"]


class LifecycleEventInput(BaseModel):
    """A single lifecycle transition appended to a strategy stream."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    event: LifecycleEventType
    rationale: str = ""


# --- Stored records (what the query API returns) -----------------------------


class Experiment(ExperimentInput):
    """A persisted experiment: input bundle + generated identity + append time."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    trial_fingerprint: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class Result(ResultInput):
    """A persisted harness result."""

    model_config = ConfigDict(frozen=True)

    result_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class Decision(DecisionInput):
    """A persisted decision."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class Strategy(StrategyInput):
    """A persisted strategy genesis record."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class StrategyVersion(StrategyVersionInput):
    """A persisted strategy version."""

    model_config = ConfigDict(frozen=True)

    strategy_version_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class LifecycleEvent(LifecycleEventInput):
    """A persisted lifecycle transition."""

    model_config = ConfigDict(frozen=True)

    lifecycle_event_id: str
    seq: int
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


# --- Lifecycle state machine -------------------------------------------------

# Legal transitions. Linear: create -> candidate -> promote -> promoted ->
# retire -> retired. "retire-before-promote" (candidate -> retire) is illegal.
_TRANSITIONS: dict[LifecycleState | None, dict[LifecycleEventType, LifecycleState]] = {
    None: {"create": "candidate"},
    "candidate": {"promote": "promoted"},
    "promoted": {"retire": "retired"},
    "retired": {},
}


class IllegalTransitionError(ValueError):
    """Raised when a lifecycle event is illegal given the current folded state."""


def next_state(current: LifecycleState | None, event: LifecycleEventType) -> LifecycleState:
    """Return the state reached by applying ``event`` to ``current``.

    Raises ``IllegalTransitionError`` for an out-of-order / illegal transition.
    """
    allowed = _TRANSITIONS.get(current, {})
    if event not in allowed:
        msg = (
            f"Illegal lifecycle transition: event {event!r} is not permitted "
            f"from state {current!r}."
        )
        raise IllegalTransitionError(msg)
    return allowed[event]


def fold_lifecycle(events: list[LifecycleEventType]) -> LifecycleState | None:
    """Fold an ordered list of lifecycle events into the current state.

    Contract: the strategy's status is *derived* from this fold, never stored as
    a mutable column. Returns ``None`` for an empty stream.
    """
    state: LifecycleState | None = None
    for event in events:
        state = next_state(state, event)
    return state


# --- Forecast taxonomy (DELPHI §3/§5) ----------------------------------------
#
# A question stream is the immutable record of one forecasting target:
#   Question (genesis) -> EvidenceSet(s) -> Forecast(s) -> Resolution(s).
# Every forecast must write a complete record (CLAUDE.md §3): the question, the
# as-of time, the evidence set with knowledge-time stamps, model/version
# provenance, the workflow trace, and — once known — the resolution.


class QuestionInput(BaseModel):
    """Immutable, normalized, resolvable question (the output of intake).

    Contract: a question is only recorded once it is *resolvable* — it carries
    explicit ``resolution_criteria`` and a ``domain`` (used for per-domain
    calibration, §2.3). Unresolvable questions are refused at intake (§10).
    """

    model_config = ConfigDict(frozen=True)

    text: str
    question_type: QuestionType
    domain: str
    resolution_criteria: str
    close_time: datetime | None = None
    source: str = ""
    metadata: dict[str, Any] = {}

    @field_validator("text", "domain", "resolution_criteria")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "Question text, domain, and resolution_criteria are required."
            raise ValueError(msg)
        return v

    @field_validator("close_time")
    @classmethod
    def _utc_close(cls, v: datetime | None) -> datetime | None:
        return None if v is None else ensure_utc(v)


class EvidenceItem(BaseModel):
    """One retrieved evidence snippet pinned at or before the forecast as-of.

    Domain-agnostic provenance record: ``knowledge_time`` is when the fact could
    first have been known. The enclosing :class:`EvidenceSetInput` enforces that
    every item's knowledge_time is ``<= as_of`` (no look-ahead, §2.1).
    """

    model_config = ConfigDict(frozen=True)

    snippet: str
    source: str
    source_id: str = ""
    knowledge_time: datetime
    score: float = Field(default=0.0, ge=0.0)
    query: str = ""

    @field_validator("snippet", "source")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "EvidenceItem snippet and source are required."
            raise ValueError(msg)
        return v

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class EvidenceSetInput(BaseModel):
    """The evidence gathered for a question as of an explicit ceiling.

    Contract: ``as_of`` is the knowledge-time ceiling and every item must be
    dated at or before it. A post-as-of item is a Prime Directive §2.1
    violation and is rejected at the boundary.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    as_of: datetime
    items: tuple[EvidenceItem, ...] = ()

    @field_validator("question_id")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "EvidenceSetInput.question_id is required."
            raise ValueError(msg)
        return v

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _items_within_as_of(self) -> Self:
        for item in self.items:
            if item.knowledge_time > self.as_of:
                msg = (
                    "Evidence item knowledge_time must be <= as_of "
                    "(no look-ahead); "
                    f"{item.knowledge_time.isoformat()} > {self.as_of.isoformat()}."
                )
                raise ValueError(msg)
        return self


class Quantile(BaseModel):
    """One point on a predictive CDF: ``value`` at cumulative probability ``level``."""

    model_config = ConfigDict(frozen=True)

    level: float = Field(gt=0.0, lt=1.0)
    value: float


class ForecastInput(BaseModel):
    """A formed forecast: a probability (binary) or a quantile set (distribution).

    Contract (CLAUDE.md §3/§9): a complete forecast carries the ``as_of`` ceiling,
    a rationale, the model/version provenance, the workflow trace, and a
    reproducibility handle sufficient to reproduce it exactly. Exactly one of
    ``probability`` or ``quantiles`` is provided.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    as_of: datetime
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    quantiles: tuple[Quantile, ...] | None = None
    rationale: str
    evidence_set_id: str | None = None
    model_provenance: dict[str, Any]
    trace: dict[str, Any] = {}
    calibration_metadata: dict[str, Any] = {}
    uncertainty: dict[str, Any] | None = None
    repro_handle: dict[str, Any]

    @field_validator("question_id", "rationale")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "ForecastInput.question_id and rationale are required."
            raise ValueError(msg)
        return v

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _validate_prediction(self) -> Self:
        has_probability = self.probability is not None
        has_quantiles = self.quantiles is not None
        if has_probability == has_quantiles:
            msg = "Provide exactly one of probability or quantiles."
            raise ValueError(msg)
        if self.quantiles is not None:
            if not self.quantiles:
                msg = "quantiles must be non-empty when provided."
                raise ValueError(msg)
            levels = [q.level for q in self.quantiles]
            if any(a >= b for a, b in zip(levels, levels[1:], strict=False)):
                msg = "quantile levels must be strictly increasing."
                raise ValueError(msg)
            values = [q.value for q in self.quantiles]
            if any(a > b for a, b in zip(values, values[1:], strict=False)):
                msg = "quantile values must be non-decreasing."
                raise ValueError(msg)
        if not self.model_provenance:
            msg = "ForecastInput.model_provenance is required (no anonymous forecasts)."
            raise ValueError(msg)
        if not self.repro_handle:
            msg = "ForecastInput.repro_handle is required (forecasts must be reproducible)."
            raise ValueError(msg)
        return self


class ResolutionInput(BaseModel):
    """Ground-truth resolution of a question once it closes.

    ``resolved_value`` is the realized outcome on the question's native scale
    (0.0/1.0 for binary, the realized number for numeric/date). ``source`` names
    where the truth came from, for provenance.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    resolved_value: float
    resolved_at: datetime
    source: str
    forecast_id: str | None = None
    resolved_label: str = ""
    notes: str = ""

    @field_validator("question_id", "source")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "ResolutionInput.question_id and source are required."
            raise ValueError(msg)
        return v

    @field_validator("resolved_at")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class Question(QuestionInput):
    """A persisted question: input bundle + generated identity + append time."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_kt(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class EvidenceSet(EvidenceSetInput):
    """A persisted evidence set."""

    model_config = ConfigDict(frozen=True)

    evidence_set_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_kt(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class Forecast(ForecastInput):
    """A persisted forecast."""

    model_config = ConfigDict(frozen=True)

    forecast_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_kt(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class Resolution(ResolutionInput):
    """A persisted resolution."""

    model_config = ConfigDict(frozen=True)

    resolution_id: str
    knowledge_time: datetime

    @field_validator("knowledge_time")
    @classmethod
    def _utc_kt(cls, v: datetime) -> datetime:
        return ensure_utc(v)
