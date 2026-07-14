"""Disciplined disagreement-resolving supervisor for forecast ensembles.

Contract: identify disagreement among ensemble draws, resolve via targeted
as-of search, and apply the update ONLY at high confidence; otherwise fall back
to the robust aggregate. Forecast layer only — never touches capital.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.llm import MalformedLLMOutput, StructuredLLMClient
from core.forecast.ensemble import EnsembleForecast
from core.forecast.search import AsOfSearcher, Evidence
from core.pit.models import ensure_utc
from core.registry.fingerprint import content_hash

SUPERVISOR_PROMPT_VERSION = "supervisor_v1"
DEFAULT_SPREAD_THRESHOLD = 0.15
DEFAULT_OUTLIER_STD_MULTIPLIER = 1.5
DEFAULT_MULTIMODAL_GAP = 0.3
DEFAULT_MIN_CLUSTER_SIZE = 2


class Confidence(StrEnum):
    """Supervisor confidence in a reconciliation proposal."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DisagreementKind(StrEnum):
    """Why the supervisor invoked resolution search."""

    SPREAD = "spread"
    OUTLIER = "outlier"
    MULTIMODAL = "multimodal"
    NONE = "none"


@dataclass(frozen=True)
class Disagreement:
    """Material disagreement summary across ensemble draws."""

    kind: DisagreementKind
    spread: float
    aggregate_probability: float
    outlier_indices: tuple[int, ...]
    cluster_low: float | None
    cluster_high: float | None
    largest_gap: float

    @property
    def material(self) -> bool:
        return self.kind != DisagreementKind.NONE


class ReconciliationProposal(BaseModel):
    """LLM proposal to resolve a disagreement — not applied until gated."""

    model_config = ConfigDict(frozen=True)

    probability: float = Field(ge=0.0, le=1.0)
    confidence: Confidence
    reasoning: str = ""


class ReconciledForecast(BaseModel):
    """Supervisor output — generic, domain-agnostic reconciled forecast."""

    model_config = ConfigDict(frozen=True)

    probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0)
    aggregate_probability: float = Field(ge=0.0, le=1.0)
    confidence: Confidence
    applied: bool
    knowledge_time: datetime
    disagreement: DisagreementKind
    trajectory: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("knowledge_time")
    @classmethod
    def _utc_only(cls, v: datetime) -> datetime:
        return ensure_utc(v)


@runtime_checkable
class SupervisorLLM(Protocol):
    """Mockable LLM seam for disagreement resolution proposals."""

    @property
    def model_version(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def propose(
        self,
        *,
        disagreement: Disagreement,
        evidence: Sequence[Evidence],
        ensemble: EnsembleForecast,
    ) -> ReconciliationProposal:
        """Propose a reconciled probability with confidence — never blend/pick-best."""
        ...


@dataclass(frozen=True)
class FixtureSupervisorResponse:
    """One scripted supervisor response for ``FixtureSupervisorLLM``."""

    probability: float
    confidence: Confidence
    reasoning: str = ""


class FixtureSupervisorLLM:
    """Deterministic supervisor LLM for tests (no network)."""

    def __init__(
        self,
        responses: Mapping[str, FixtureSupervisorResponse] | None = None,
        *,
        default: FixtureSupervisorResponse | None = None,
        model_version: str = "fixture-supervisor-v1",
        prompt_version: str = SUPERVISOR_PROMPT_VERSION,
    ) -> None:
        self._responses = dict(responses or {})
        self._default = default or FixtureSupervisorResponse(
            probability=0.5,
            confidence=Confidence.LOW,
            reasoning="fixture default",
        )
        self._model_version = model_version
        self._prompt_version = prompt_version
        self.call_count = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def propose(
        self,
        *,
        disagreement: Disagreement,
        evidence: Sequence[Evidence],
        ensemble: EnsembleForecast,
    ) -> ReconciliationProposal:
        del evidence, ensemble
        self.call_count += 1
        key = disagreement.kind.value
        response = self._responses.get(key, self._default)
        return ReconciliationProposal(
            probability=response.probability,
            confidence=response.confidence,
            reasoning=response.reasoning,
        )


# The supervisor proposes a single reconciled probability with a confidence
# grade; the caller applies it ONLY at HIGH confidence (never blends/picks-best).
_SUPERVISOR_SYSTEM = (
    "You are a supervising forecaster resolving disagreement among an ensemble "
    "of independent probability estimates for a binary event. Using the "
    "disagreement summary and any targeted evidence, propose a single reconciled "
    "probability and grade your confidence. Respond with ONLY a JSON object of "
    'the form {"probability": p, "confidence": "high"|"medium"|"low", '
    '"reasoning": "..."} where p is in [0, 1]. Reserve "high" for genuinely '
    "decisive evidence. Do not include any prose outside the JSON object."
)

_MAX_SUPERVISOR_EVIDENCE = 8
_SUPERVISOR_SNIPPET_MAX = 300


def _compose_supervisor_user(
    *,
    disagreement: Disagreement,
    evidence: Sequence[Evidence],
    ensemble: EnsembleForecast,
) -> str:
    """Render the disagreement, ensemble, and evidence into a proposal prompt."""
    lines = [
        f"Disagreement kind: {disagreement.kind.value}",
        f"Ensemble aggregate probability: {disagreement.aggregate_probability:.4f}",
        f"Ensemble spread: {disagreement.spread:.4f}",
        f"Ensemble n: {ensemble.n}",
    ]
    if disagreement.cluster_low is not None and disagreement.cluster_high is not None:
        lines.append(
            f"Bimodal clusters near {disagreement.cluster_low:.3f} and "
            f"{disagreement.cluster_high:.3f}"
        )
    if evidence:
        lines.append("\nTargeted evidence:")
        for ev in evidence[:_MAX_SUPERVISOR_EVIDENCE]:
            snippet = ev.snippet[:_SUPERVISOR_SNIPPET_MAX]
            lines.append(f"- [{ev.source}:{ev.source_id}] {snippet}")
    else:
        lines.append("\nNo additional evidence retrieved.")
    return "\n".join(lines)


def _coerce_proposal(payload: Mapping[str, Any]) -> ReconciliationProposal:
    """Validate a parsed payload into a ``ReconciliationProposal``."""
    if "probability" not in payload:
        msg = f"supervisor payload missing 'probability' key: {payload!r}"
        raise MalformedLLMOutput(msg)
    try:
        probability = float(payload["probability"])
    except (TypeError, ValueError) as exc:
        msg = f"supervisor 'probability' is not a number: {payload['probability']!r}"
        raise MalformedLLMOutput(msg) from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        msg = f"supervisor 'probability' out of [0, 1]: {probability!r}"
        raise MalformedLLMOutput(msg)
    raw_conf = payload.get("confidence")
    try:
        confidence = Confidence(str(raw_conf).lower())
    except ValueError as exc:
        msg = f"supervisor 'confidence' not one of high/medium/low: {raw_conf!r}"
        raise MalformedLLMOutput(msg) from exc
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        msg = f"supervisor 'reasoning' must be a string: {reasoning!r}"
        raise MalformedLLMOutput(msg)
    return ReconciliationProposal(
        probability=probability,
        confidence=confidence,
        reasoning=reasoning,
    )


class BedrockSupervisorLLM:
    """Structured-LLM-backed ``SupervisorLLM``: proposes a gated reconciled probability.

    Implements the ``SupervisorLLM`` protocol over a shared ``StructuredLLMClient``
    (the direct Anthropic API by default, or Bedrock). Forecast layer only — never
    touches the capital path (CLAUDE.md section 10). The proposal is advisory until
    the caller's HIGH-confidence gate.
    """

    def __init__(
        self,
        client: StructuredLLMClient,
        *,
        prompt_version: str = SUPERVISOR_PROMPT_VERSION,
        system: str = _SUPERVISOR_SYSTEM,
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

    def propose(
        self,
        *,
        disagreement: Disagreement,
        evidence: Sequence[Evidence],
        ensemble: EnsembleForecast,
    ) -> ReconciliationProposal:
        payload = self._client.invoke_structured(
            system=self._system,
            user=_compose_supervisor_user(
                disagreement=disagreement,
                evidence=evidence,
                ensemble=ensemble,
            ),
        )
        return _coerce_proposal(payload)


@dataclass(frozen=True)
class ReconciliationCacheKey:
    """Content-addressed reconciliation cache key."""

    ensemble_fingerprint: str
    supervisor_model_version: str
    supervisor_prompt_version: str
    search_config: str


class ReconciliationCache(ABC):
    """Append-only reconciliation cache — identical keys are idempotent."""

    @abstractmethod
    def get(self, key: ReconciliationCacheKey) -> ReconciledForecast | None:
        """Return cached reconciliation for an addressing key, or None on miss."""

    @abstractmethod
    def put(self, key: ReconciliationCacheKey, forecast: ReconciledForecast) -> None:
        """Append cache entry; identical keys are idempotent."""


class InMemoryReconciliationCache(ReconciliationCache):
    """In-memory reconciliation cache for tests and local development."""

    def __init__(self) -> None:
        self._entries: dict[ReconciliationCacheKey, ReconciledForecast] = {}

    def get(self, key: ReconciliationCacheKey) -> ReconciledForecast | None:
        return self._entries.get(key)

    def put(self, key: ReconciliationCacheKey, forecast: ReconciledForecast) -> None:
        if key in self._entries:
            return
        self._entries[key] = forecast

    @property
    def keys(self) -> tuple[ReconciliationCacheKey, ...]:
        return tuple(self._entries.keys())


def build_supervisor_config(
    *,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
    outlier_std_multiplier: float = DEFAULT_OUTLIER_STD_MULTIPLIER,
    multimodal_gap: float = DEFAULT_MULTIMODAL_GAP,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> str:
    """Serialize supervisor parameters for content-addressed cache keys."""
    return (
        f"spread={spread_threshold}|outlier={outlier_std_multiplier}|"
        f"gap={multimodal_gap}|cluster={min_cluster_size}"
    )


def _ensemble_fingerprint(ensemble: EnsembleForecast) -> str:
    payload = {
        "probability": ensemble.probability,
        "uncertainty": ensemble.uncertainty,
        "n": ensemble.n,
        "aggregator": ensemble.aggregator,
        "trim_fraction": ensemble.trim_fraction,
        "knowledge_time": ensemble.knowledge_time.isoformat(),
        "draws": [d.model_dump(mode="json") for d in ensemble.draws],
        "provenance": dict(ensemble.provenance),
    }
    return content_hash(payload)


def detect_disagreement(
    ensemble: EnsembleForecast,
    *,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
    outlier_std_multiplier: float = DEFAULT_OUTLIER_STD_MULTIPLIER,
    multimodal_gap: float = DEFAULT_MULTIMODAL_GAP,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> Disagreement:
    """Detect material disagreement from the draw probability distribution."""
    if spread_threshold < 0.0:
        msg = "spread_threshold must be non-negative"
        raise ValueError(msg)
    if outlier_std_multiplier < 0.0:
        msg = "outlier_std_multiplier must be non-negative"
        raise ValueError(msg)
    if multimodal_gap < 0.0:
        msg = "multimodal_gap must be non-negative"
        raise ValueError(msg)
    if min_cluster_size < 1:
        msg = "min_cluster_size must be >= 1"
        raise ValueError(msg)

    probabilities = [d.probability for d in ensemble.draws]
    spread = ensemble.uncertainty
    aggregate = ensemble.probability

    outlier_indices: list[int] = []
    if spread > 0.0 and outlier_std_multiplier > 0.0:
        threshold = outlier_std_multiplier * spread
        outlier_indices = [i for i, p in enumerate(probabilities) if abs(p - aggregate) > threshold]

    sorted_probs = sorted(probabilities)
    largest_gap = 0.0
    gap_index = -1
    for i in range(len(sorted_probs) - 1):
        gap = sorted_probs[i + 1] - sorted_probs[i]
        if gap > largest_gap:
            largest_gap = gap
            gap_index = i

    cluster_low: float | None = None
    cluster_high: float | None = None
    multimodal = False
    if gap_index >= 0 and largest_gap >= multimodal_gap:
        low_cluster = sorted_probs[: gap_index + 1]
        high_cluster = sorted_probs[gap_index + 1 :]
        if len(low_cluster) >= min_cluster_size and len(high_cluster) >= min_cluster_size:
            multimodal = True
            cluster_low = low_cluster[-1]
            cluster_high = high_cluster[0]

    if multimodal:
        kind = DisagreementKind.MULTIMODAL
    elif outlier_indices:
        kind = DisagreementKind.OUTLIER
    elif spread > spread_threshold:
        kind = DisagreementKind.SPREAD
    else:
        kind = DisagreementKind.NONE

    return Disagreement(
        kind=kind,
        spread=spread,
        aggregate_probability=aggregate,
        outlier_indices=tuple(outlier_indices),
        cluster_low=cluster_low,
        cluster_high=cluster_high,
        largest_gap=largest_gap,
    )


def build_resolution_query(
    ensemble: EnsembleForecast,
    disagreement: Disagreement,
) -> str:
    """Build a targeted as-of query aimed at the specific disagreement."""
    provenance = dict(ensemble.provenance)
    question = provenance.get("question")
    if isinstance(question, str) and question.strip():
        base = question.strip()
    else:
        base = "forecast reference class"

    if disagreement.kind == DisagreementKind.MULTIMODAL:
        return (
            f"{base} resolve bimodal disagreement "
            f"clusters {disagreement.cluster_low:.2f}/{disagreement.cluster_high:.2f}"
        )
    if disagreement.kind == DisagreementKind.OUTLIER and disagreement.outlier_indices:
        outlier_rationales: list[str] = []
        for idx in disagreement.outlier_indices:
            draw = ensemble.draws[idx]
            prov = dict(draw.provenance)
            rationale = prov.get("rationale") or prov.get("reference_class")
            if isinstance(rationale, str) and rationale.strip():
                outlier_rationales.append(rationale.strip())
        if outlier_rationales:
            joined = "; ".join(outlier_rationales[:3])
            return f"{base} outlier evidence {joined}"
        return f"{base} outlier runs at indices {list(disagreement.outlier_indices)}"
    if disagreement.kind == DisagreementKind.SPREAD:
        return f"{base} high ensemble spread {disagreement.spread:.3f}"
    return base


def _assert_evidence_as_of(evidence: Sequence[Evidence], *, as_of: datetime) -> None:
    as_of = ensure_utc(as_of)
    for item in evidence:
        if item.knowledge_time > as_of:
            msg = "as_of_search returned evidence with knowledge_time > as_of"
            raise RuntimeError(msg)


def _fallback_forecast(
    ensemble: EnsembleForecast,
    disagreement: Disagreement,
    *,
    trajectory: dict[str, Any],
    extra_provenance: Mapping[str, Any] | None = None,
) -> ReconciledForecast:
    provenance: dict[str, Any] = {
        "supervisor_method": "disagreement_targeted_search",
        "applied": False,
        "fallback_reason": "confidence_below_high_or_no_disagreement",
        "aggregate_probability": ensemble.probability,
        "disagreement_kind": disagreement.kind.value,
        "disagreement_spread": disagreement.spread,
    }
    if extra_provenance:
        provenance.update(dict(extra_provenance))
    return ReconciledForecast(
        probability=ensemble.probability,
        uncertainty=ensemble.uncertainty,
        aggregate_probability=ensemble.probability,
        confidence=Confidence.LOW,
        applied=False,
        knowledge_time=ensemble.knowledge_time,
        disagreement=disagreement.kind,
        trajectory=trajectory,
        provenance=provenance,
    )


class Supervisor:
    """Disciplined agentic supervisor with confidence gate and aggregate fallback."""

    def __init__(
        self,
        search: AsOfSearcher,
        llm: SupervisorLLM,
        cache: ReconciliationCache,
        *,
        spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
        outlier_std_multiplier: float = DEFAULT_OUTLIER_STD_MULTIPLIER,
        multimodal_gap: float = DEFAULT_MULTIMODAL_GAP,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    ) -> None:
        self._search = search
        self._llm = llm
        self._cache = cache
        self._spread_threshold = spread_threshold
        self._outlier_std_multiplier = outlier_std_multiplier
        self._multimodal_gap = multimodal_gap
        self._min_cluster_size = min_cluster_size
        self.search_call_count = 0

    def reconcile(self, ensemble: EnsembleForecast) -> ReconciledForecast:
        """Identify disagreements, resolve with targeted as-of search, gate on confidence.

        Returns a reconciled forecast ONLY if confidence is high; otherwise returns
        the robust aggregate unchanged. The result can improve on or equal the
        aggregate — never underperform it. Forecast layer only; never touches capital.
        """
        disagreement = detect_disagreement(
            ensemble,
            spread_threshold=self._spread_threshold,
            outlier_std_multiplier=self._outlier_std_multiplier,
            multimodal_gap=self._multimodal_gap,
            min_cluster_size=self._min_cluster_size,
        )

        if not disagreement.material:
            trajectory = {
                "skipped": True,
                "reason": "no_material_disagreement",
                "disagreement": disagreement.kind.value,
            }
            return _fallback_forecast(ensemble, disagreement, trajectory=trajectory)

        fingerprint = _ensemble_fingerprint(ensemble)
        config = build_supervisor_config(
            spread_threshold=self._spread_threshold,
            outlier_std_multiplier=self._outlier_std_multiplier,
            multimodal_gap=self._multimodal_gap,
            min_cluster_size=self._min_cluster_size,
        )
        cache_key = ReconciliationCacheKey(
            ensemble_fingerprint=fingerprint,
            supervisor_model_version=self._llm.model_version,
            supervisor_prompt_version=self._llm.prompt_version,
            search_config=config,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(
                update={
                    "provenance": {
                        **dict(cached.provenance),
                        "cached": True,
                    }
                }
            )

        query = build_resolution_query(ensemble, disagreement)
        as_of = ensure_utc(ensemble.knowledge_time)
        self.search_call_count += 1
        evidence = self._search.as_of_search(query, as_of=as_of)
        _assert_evidence_as_of(evidence, as_of=as_of)

        proposal = self._llm.propose(
            disagreement=disagreement,
            evidence=evidence,
            ensemble=ensemble,
        )

        trajectory = {
            "skipped": False,
            "query": query,
            "as_of": as_of.isoformat(),
            "disagreement": {
                "kind": disagreement.kind.value,
                "spread": disagreement.spread,
                "outlier_indices": list(disagreement.outlier_indices),
                "cluster_low": disagreement.cluster_low,
                "cluster_high": disagreement.cluster_high,
                "largest_gap": disagreement.largest_gap,
            },
            "evidence": [ev.model_dump(mode="json") for ev in evidence],
            "proposal": proposal.model_dump(mode="json"),
        }

        if proposal.confidence != Confidence.HIGH:
            result = _fallback_forecast(
                ensemble,
                disagreement,
                trajectory=trajectory,
                extra_provenance={
                    "proposal_probability": proposal.probability,
                    "proposal_confidence": proposal.confidence.value,
                    "proposal_reasoning": proposal.reasoning,
                },
            )
            self._cache.put(cache_key, result)
            return result

        result = ReconciledForecast(
            probability=proposal.probability,
            uncertainty=ensemble.uncertainty,
            aggregate_probability=ensemble.probability,
            confidence=proposal.confidence,
            applied=True,
            knowledge_time=ensemble.knowledge_time,
            disagreement=disagreement.kind,
            trajectory=trajectory,
            provenance={
                "supervisor_method": "disagreement_targeted_search",
                "applied": True,
                "aggregate_probability": ensemble.probability,
                "proposal_probability": proposal.probability,
                "proposal_confidence": proposal.confidence.value,
                "proposal_reasoning": proposal.reasoning,
                "disagreement_kind": disagreement.kind.value,
                "disagreement_spread": disagreement.spread,
                "cached": False,
            },
        )
        self._cache.put(cache_key, result)
        return result


def reconciliation_to_audit_dict(forecast: ReconciledForecast) -> str:
    """Serialize a reconciliation for reproducibility checks."""
    return json.dumps(
        forecast.model_dump(mode="json"),
        sort_keys=True,
        default=str,
    )


__all__ = [
    "Confidence",
    "DEFAULT_MIN_CLUSTER_SIZE",
    "DEFAULT_MULTIMODAL_GAP",
    "DEFAULT_OUTLIER_STD_MULTIPLIER",
    "DEFAULT_SPREAD_THRESHOLD",
    "Disagreement",
    "DisagreementKind",
    "FixtureSupervisorLLM",
    "FixtureSupervisorResponse",
    "InMemoryReconciliationCache",
    "ReconciledForecast",
    "ReconciliationCache",
    "ReconciliationCacheKey",
    "ReconciliationProposal",
    "SUPERVISOR_PROMPT_VERSION",
    "Supervisor",
    "SupervisorLLM",
    "build_resolution_query",
    "build_supervisor_config",
    "detect_disagreement",
    "reconciliation_to_audit_dict",
]
