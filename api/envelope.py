"""Response envelope (C10.3) — everything a buyer needs to trust the number.

Assembles the §9 response surface from a completed forecast: the calibrated
probability, the rationale (decomposition, reference classes, red-team counter),
evidence provenance *with knowledge-times*, calibration metadata + an honest
confidence band, the resolution criteria as DELPHI understood them, and a
reproducibility handle. We hide the *how* (routing) but expose the *why* — that
asymmetry is the product (§9). A refused question yields a refusal envelope.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.registry.store import RegistryStore
from forecaster.chain import ForecastResult

__all__ = ["ConfidenceBand", "EvidenceProvenance", "ForecastEnvelope", "build_envelope"]


class EvidenceProvenance(BaseModel):
    """One evidence item with its knowledge-time (the trust surface, §9)."""

    model_config = ConfigDict(frozen=True)

    source: str
    source_id: str
    knowledge_time: datetime
    snippet: str
    score: float


class ConfidenceBand(BaseModel):
    """Honest confidence band around the reported probability."""

    model_config = ConfigDict(frozen=True)

    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)


class ForecastEnvelope(BaseModel):
    """The full §9 response envelope (or a refusal)."""

    model_config = ConfigDict(frozen=True)

    refused: bool
    refusal_reason: str = ""
    probability: float | None = None
    confidence_band: ConfidenceBand | None = None
    rationale: str = ""
    red_team_counter: str = ""
    evidence: tuple[EvidenceProvenance, ...] = ()
    calibration_metadata: dict[str, Any] = Field(default_factory=dict)
    resolution_criteria: str = ""
    reproducibility_handle: dict[str, Any] = Field(default_factory=dict)
    workflow: dict[str, Any] | None = None
    retained: bool = True
    providers: tuple[str, ...] = ()


def build_envelope(
    result: ForecastResult,
    *,
    store: RegistryStore,
    red_team_counter: str = "",
    workflow: dict[str, Any] | None = None,
    retained: bool = True,
    providers: tuple[str, ...] = (),
) -> ForecastEnvelope:
    """Build the response envelope from a completed (or refused) forecast."""
    if not result.accepted or result.probability is None or result.question_id is None:
        reason = ""
        if result.refusal is not None and result.refusal.reason is not None:
            reason = result.refusal.reason.value
        return ForecastEnvelope(
            refused=True,
            refusal_reason=reason,
            retained=retained,
            providers=providers,
        )

    forecast = store.forecasts_for(result.question_id)[-1]
    question = store.get_question(result.question_id)
    band = result.uncertainty.combined if result.uncertainty is not None else 0.0
    confidence = ConfidenceBand(
        low=max(0.0, result.probability - band),
        high=min(1.0, result.probability + band),
    )
    evidence = tuple(
        EvidenceProvenance(
            source=ev.source,
            source_id=ev.source_id,
            knowledge_time=ev.knowledge_time,
            snippet=ev.snippet,
            score=ev.score,
        )
        for ev in result.evidence
    )
    return ForecastEnvelope(
        refused=False,
        probability=result.probability,
        confidence_band=confidence,
        rationale=result.rationale,
        red_team_counter=red_team_counter,
        evidence=evidence,
        calibration_metadata=dict(forecast.calibration_metadata),
        resolution_criteria=question.resolution_criteria,
        reproducibility_handle=dict(forecast.repro_handle),
        workflow=workflow,
        retained=retained,
        providers=providers,
    )
