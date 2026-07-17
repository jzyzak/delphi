"""System-wide defense-in-depth leakage judge for forecast traces.

The structural PIT guarantee is primary. This module audits evidence and
reasoning traces for post-as_of references that could indicate misdated
documents, hallucinated future facts, or corpus bugs. Tune for high recall;
unflagged traces are treated as reliably clean.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.llm import MalformedLLMOutput, StructuredLLMClient
from core.forecast.ensemble import EnsembleForecast
from core.forecast.search import Evidence
from core.forecast.supervisor import ReconciledForecast
from core.pit.models import ensure_utc

_LOG = structlog.get_logger(__name__)
LEAKAGE_JUDGE_PROMPT_VERSION = "leakage_judge_v1"
DEFAULT_TRACE_EXCERPT_LEN = 500


class TraceComponent(StrEnum):
    """Forecast pipeline component that produced a trace."""

    EXTRACTION = "extraction"
    ENSEMBLE = "ensemble"
    SEARCH = "search"
    SUPERVISOR = "supervisor"


class Trace(BaseModel):
    """One auditable evidence/reasoning trace pinned at a knowledge-time."""

    model_config = ConfigDict(frozen=True)

    component: TraceComponent
    as_of: datetime
    text: str
    forecast_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class LeakageVerdict(BaseModel):
    """Outcome of a high-recall leakage audit."""

    model_config = ConfigDict(frozen=True)

    flagged: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    component: TraceComponent
    trace_excerpt: str = ""


class RegistrySlice(BaseModel):
    """Batch audit input: a tagged collection of forecast traces."""

    model_config = ConfigDict(frozen=True)

    traces: tuple[Trace, ...] = Field(default_factory=tuple)
    slice_id: str = ""


class ComponentLeakageRate(BaseModel):
    """Leakage-rate estimate for one pipeline component."""

    model_config = ConfigDict(frozen=True)

    component: TraceComponent
    total: int = Field(ge=0)
    flagged: int = Field(ge=0)
    rate: float = Field(ge=0.0, le=1.0)


class LeakageRegression(BaseModel):
    """A per-component leakage-rate spike vs a baseline."""

    model_config = ConfigDict(frozen=True)

    component: TraceComponent
    current_rate: float
    baseline_rate: float
    delta: float


class LeakageReport(BaseModel):
    """Batch leakage-rate estimate with optional regression surfacing."""

    model_config = ConfigDict(frozen=True)

    slice_id: str = ""
    total: int = Field(ge=0)
    flagged: int = Field(ge=0)
    aggregate_rate: float = Field(ge=0.0, le=1.0)
    by_component: tuple[ComponentLeakageRate, ...] = Field(default_factory=tuple)
    regressions: tuple[LeakageRegression, ...] = Field(default_factory=tuple)


class QuarantineDisposition(StrEnum):
    """Disposition states — only PENDING is set by the judge."""

    PENDING = "pending"


class QuarantineRecord(BaseModel):
    """Conservative quarantine log for the Research Director (17) to decide."""

    model_config = ConfigDict(frozen=True)

    forecast_id: str
    component: TraceComponent
    as_of: datetime
    verdict: LeakageVerdict
    disposition: QuarantineDisposition = QuarantineDisposition.PENDING

    @field_validator("as_of")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


@runtime_checkable
class LeakageJudgeLLM(Protocol):
    """Mockable LLM seam for post-as_of leakage detection."""

    @property
    def model_version(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def audit(self, trace: str, *, as_of: datetime) -> LeakageVerdict:
        """Flag any post-as_of reference in ``trace`` relative to ``as_of``."""
        ...


class FixtureLeakageJudgeLLM:
    """Deterministic high-recall leakage judge for tests (no network).

    Flags when any configured substring appears in the trace. Optionally rejects
    ISO timestamps in the trace that are strictly after ``as_of``.
    """

    def __init__(
        self,
        *,
        flag_substrings: tuple[str, ...] = (),
        reject_future_iso_dates: bool = True,
        model_version: str = "fixture-leakage-judge-v1",
        prompt_version: str = LEAKAGE_JUDGE_PROMPT_VERSION,
        flagged_confidence: float = 0.95,
        clean_confidence: float = 0.99,
    ) -> None:
        self._flag_substrings = flag_substrings
        self._reject_future_iso_dates = reject_future_iso_dates
        self._model_version = model_version
        self._prompt_version = prompt_version
        self._flagged_confidence = flagged_confidence
        self._clean_confidence = clean_confidence
        self.call_count = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def audit(self, trace: str, *, as_of: datetime) -> LeakageVerdict:
        self.call_count += 1
        as_of = ensure_utc(as_of)
        reasons: list[str] = []
        for needle in self._flag_substrings:
            if needle in trace:
                reasons.append(f"substring:{needle}")

        if self._reject_future_iso_dates:
            for token in trace.split():
                cleaned = token.strip(",\"'[]{}")
                if "T" not in cleaned or len(cleaned) < 10:
                    continue
                try:
                    parsed = ensure_utc(datetime.fromisoformat(cleaned.replace("Z", "+00:00")))
                except ValueError:
                    continue
                if parsed > as_of:
                    reasons.append(f"future_timestamp:{cleaned}")

        flagged = len(reasons) > 0
        rationale = "; ".join(reasons) if reasons else "no post-as_of references detected"
        excerpt = (
            trace[:DEFAULT_TRACE_EXCERPT_LEN] if len(trace) > DEFAULT_TRACE_EXCERPT_LEN else trace
        )
        return LeakageVerdict(
            flagged=flagged,
            confidence=self._flagged_confidence if flagged else self._clean_confidence,
            rationale=rationale,
            component=TraceComponent.SEARCH,
            trace_excerpt=excerpt,
        )


# High-recall by design: prefer false positives (flag-and-quarantine) over
# missing a genuine post-as_of reference. The ``component`` on the returned
# verdict is a placeholder; the consumer (``LeakageJudge.audit``) overwrites it
# from the originating ``Trace``.
_LEAKAGE_SYSTEM = (
    "You are a strict data-leakage auditor. You are given a reasoning/evidence "
    "trace and an as-of date. Flag the trace if it references ANY information, "
    "event, or dated fact that would only be known AFTER the as-of date. Favor "
    "recall: when in doubt, flag. Respond with ONLY a JSON object of the form "
    '{"flagged": true|false, "confidence": c, "rationale": "..."} where c is in '
    "[0, 1]. Do not include any prose outside the JSON object."
)


def _compose_leakage_user(trace: str, *, as_of: datetime) -> str:
    """Render the trace and as-of ceiling into an audit prompt."""
    return f"As-of date (knowledge-time ceiling): {as_of.isoformat()}\n\nTrace:\n{trace}"


def _coerce_leakage_fields(payload: Mapping[str, Any]) -> tuple[bool, float, str]:
    """Validate and extract (flagged, confidence, rationale) from a payload."""
    if "flagged" not in payload:
        msg = f"leakage payload missing 'flagged' key: {payload!r}"
        raise MalformedLLMOutput(msg)
    flagged = payload["flagged"]
    if not isinstance(flagged, bool):
        msg = f"leakage 'flagged' must be a boolean: {flagged!r}"
        raise MalformedLLMOutput(msg)
    if "confidence" not in payload:
        msg = f"leakage payload missing 'confidence' key: {payload!r}"
        raise MalformedLLMOutput(msg)
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        msg = f"leakage 'confidence' is not a number: {payload['confidence']!r}"
        raise MalformedLLMOutput(msg) from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        msg = f"leakage 'confidence' out of [0, 1]: {confidence!r}"
        raise MalformedLLMOutput(msg)
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        msg = f"leakage 'rationale' must be a string: {rationale!r}"
        raise MalformedLLMOutput(msg)
    return flagged, confidence, rationale


class BedrockLeakageJudgeLLM:
    """Structured-LLM-backed ``LeakageJudgeLLM``: high-recall post-as_of leakage audit.

    Implements the ``LeakageJudgeLLM`` protocol over a shared ``StructuredLLMClient``
    (the direct Anthropic API by default, or Bedrock). Defense-in-depth on top of
    the structural PIT guarantee (CLAUDE.md section 10) — it never replaces it.
    """

    def __init__(
        self,
        client: StructuredLLMClient,
        *,
        prompt_version: str = LEAKAGE_JUDGE_PROMPT_VERSION,
        system: str = _LEAKAGE_SYSTEM,
    ) -> None:
        self._client = client
        self._prompt_version = prompt_version
        self._system = system

    @property
    def model_version(self) -> str:
        return self._client.model_id

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def audit(self, trace: str, *, as_of: datetime) -> LeakageVerdict:
        as_of = ensure_utc(as_of)
        payload = self._client.invoke_structured(
            system=self._system,
            user=_compose_leakage_user(trace, as_of=as_of),
        )
        flagged, confidence, rationale = _coerce_leakage_fields(payload)
        excerpt = (
            trace[:DEFAULT_TRACE_EXCERPT_LEN] if len(trace) > DEFAULT_TRACE_EXCERPT_LEN else trace
        )
        return LeakageVerdict(
            flagged=flagged,
            confidence=confidence,
            rationale=rationale,
            component=TraceComponent.SEARCH,
            trace_excerpt=excerpt,
        )


def render_trajectory(trajectory: Mapping[str, Any]) -> str:
    """Serialize a trajectory dict into an auditable trace string."""
    return json.dumps(dict(trajectory), sort_keys=True, default=str)


def trace_from_ensemble(
    ensemble: EnsembleForecast,
    *,
    forecast_id: str = "",
) -> Trace:
    """Build an auditable trace from an ensemble forecast."""
    payload = {
        "probability": ensemble.probability,
        "uncertainty": ensemble.uncertainty,
        "n": ensemble.n,
        "aggregator": ensemble.aggregator,
        "knowledge_time": ensemble.knowledge_time.isoformat(),
        "draws": [d.model_dump(mode="json") for d in ensemble.draws],
        "provenance": dict(ensemble.provenance),
    }
    return Trace(
        component=TraceComponent.ENSEMBLE,
        as_of=ensemble.knowledge_time,
        text=json.dumps(payload, sort_keys=True, default=str),
        forecast_id=forecast_id,
        metadata={"n": ensemble.n, "aggregator": ensemble.aggregator},
    )


def trace_from_evidence(
    evidence: Sequence[Evidence],
    *,
    as_of: datetime,
    forecast_id: str = "",
) -> Trace:
    """Build an auditable SEARCH trace over the retrieved evidence snippets.

    The raw snippets are exactly where a retrieval-side leak lands (a misdated
    document, a live-fetched extract carrying post-as-of edits), so the judge
    must read them directly — auditing only downstream summaries leaves the
    weakest retrieval path uncovered.
    """
    payload = [
        {
            "source": item.source,
            "source_id": item.source_id,
            "knowledge_time": item.knowledge_time.isoformat(),
            "score": item.score,
            "query": item.query,
            "snippet": item.snippet,
        }
        for item in evidence
    ]
    return Trace(
        component=TraceComponent.SEARCH,
        as_of=as_of,
        text=json.dumps(payload, sort_keys=True, default=str),
        forecast_id=forecast_id,
        metadata={"n_evidence": len(payload)},
    )


def trace_from_supervisor(
    forecast: ReconciledForecast,
    *,
    forecast_id: str = "",
) -> Trace:
    """Build an auditable trace from a supervisor reconciliation."""
    payload = {
        "probability": forecast.probability,
        "applied": forecast.applied,
        "confidence": forecast.confidence.value,
        "knowledge_time": forecast.knowledge_time.isoformat(),
        "trajectory": dict(forecast.trajectory),
        "provenance": dict(forecast.provenance),
    }
    return Trace(
        component=TraceComponent.SUPERVISOR,
        as_of=forecast.knowledge_time,
        text=json.dumps(payload, sort_keys=True, default=str),
        forecast_id=forecast_id,
        metadata={"applied": forecast.applied},
    )


class LeakageJudge:
    """High-recall system-wide leakage judge — defense-in-depth over PIT."""

    def __init__(self, llm: LeakageJudgeLLM) -> None:
        self._llm = llm

    @property
    def model_version(self) -> str:
        return self._llm.model_version

    @property
    def prompt_version(self) -> str:
        return self._llm.prompt_version

    def audit(self, trace: Trace, *, as_of: datetime | None = None) -> LeakageVerdict:
        """Audit a trace for post-as_of leakage.

        Contract: high recall — a flag is never downgraded by low confidence.
        Defense-in-depth over the structural PIT guarantee; never a replacement.
        """
        as_of = ensure_utc(as_of or trace.as_of)
        raw = self._llm.audit(trace.text, as_of=as_of)
        return raw.model_copy(
            update={
                "component": trace.component,
                "trace_excerpt": raw.trace_excerpt or _excerpt(trace.text),
            }
        )

    def estimate_leakage_rate(
        self,
        slice: RegistrySlice,
        *,
        baseline: LeakageReport | None = None,
        spike_threshold: float = 0.05,
    ) -> LeakageReport:
        """Batch audit a slice; report per-component and aggregate leakage rates."""
        from core.forecast.leakage_batch import estimate_leakage_rate as _batch_estimate

        return _batch_estimate(
            self,
            slice,
            baseline=baseline,
            spike_threshold=spike_threshold,
        )


def audit_and_quarantine(
    judge: LeakageJudge,
    trace: Trace,
    *,
    as_of: datetime | None = None,
) -> tuple[LeakageVerdict, QuarantineRecord | None]:
    """Inline audit with conservative quarantine when flagged.

    The judge flags and quarantines; disposition is logged for the Research
    Director (17) — the judge never decides re-extract / exclude / override.
    """
    verdict = judge.audit(trace, as_of=as_of)
    if not verdict.flagged:
        return verdict, None

    record = QuarantineRecord(
        forecast_id=trace.forecast_id or trace.metadata.get("forecast_id", ""),
        component=trace.component,
        as_of=ensure_utc(as_of or trace.as_of),
        verdict=verdict,
        disposition=QuarantineDisposition.PENDING,
    )
    _LOG.warning(
        "forecast_quarantined",
        forecast_id=record.forecast_id,
        component=record.component.value,
        as_of=record.as_of.isoformat(),
        rationale=verdict.rationale,
        disposition=record.disposition.value,
    )
    return verdict, record


def _excerpt(text: str, *, max_len: int = DEFAULT_TRACE_EXCERPT_LEN) -> str:
    return text[:max_len] if len(text) > max_len else text


__all__ = [
    "DEFAULT_TRACE_EXCERPT_LEN",
    "ComponentLeakageRate",
    "FixtureLeakageJudgeLLM",
    "LEAKAGE_JUDGE_PROMPT_VERSION",
    "LeakageJudge",
    "LeakageJudgeLLM",
    "LeakageRegression",
    "LeakageReport",
    "LeakageVerdict",
    "QuarantineDisposition",
    "QuarantineRecord",
    "RegistrySlice",
    "Trace",
    "TraceComponent",
    "audit_and_quarantine",
    "render_trajectory",
    "trace_from_ensemble",
    "trace_from_evidence",
    "trace_from_supervisor",
]
