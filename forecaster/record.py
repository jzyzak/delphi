"""Registry record assembly for the forecast chain (C4.7.b).

Turns the in-flight stage outputs into the immutable registry inputs — the
EvidenceSet (with per-item knowledge-times) and the Forecast (probability,
rationale, full trace, model/version provenance, calibration metadata,
uncertainty, and a reproducibility handle). Every forecast must write a complete
record (CLAUDE.md §3); these builders are how that record is shaped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from typing import Any

from core.forecast.calibration import CalibratedForecast
from core.forecast.ensemble import EnsembleForecast
from core.forecast.search import Evidence
from core.forecast.supervisor import ReconciledForecast
from core.forecast.uncertainty import Uncertainty
from core.registry.models import EvidenceItem, EvidenceSetInput, ForecastInput
from forecaster.stages.base_rate import BaseRateEstimate
from forecaster.stages.decompose import Decomposition
from forecaster.stages.leakage_gate import LeakageGateResult

__all__ = [
    "build_evidence_set_input",
    "build_forecast_input",
    "build_rationale",
    "evidence_items",
]


def evidence_items(evidence: Sequence[Evidence]) -> tuple[EvidenceItem, ...]:
    """Map retrieved evidence into registry evidence items (knowledge-time kept)."""
    return tuple(
        EvidenceItem(
            snippet=ev.snippet,
            source=ev.source,
            source_id=ev.source_id,
            knowledge_time=ev.knowledge_time,
            score=ev.score,
            query=ev.query,
        )
        for ev in evidence
    )


def build_evidence_set_input(
    question_id: str, as_of: datetime, evidence: Sequence[Evidence]
) -> EvidenceSetInput:
    """Build the as-of EvidenceSet input for the registry."""
    return EvidenceSetInput(question_id=question_id, as_of=as_of, items=evidence_items(evidence))


def build_rationale(
    base_rate: BaseRateEstimate,
    decomposition: Decomposition,
    reconciled: ReconciledForecast,
    calibrated: CalibratedForecast,
) -> str:
    """Compose a human-readable rationale (decomposition, reference class, path)."""
    parts = [
        f"Reference class: {base_rate.reference_class} (base rate {base_rate.prior:.3f}).",
    ]
    if base_rate.rationale:
        parts.append(base_rate.rationale)
    if decomposition.sub_questions:
        subs = "; ".join(s.text for s in decomposition.sub_questions)
        parts.append(f"Decomposed ({decomposition.rule}) into: {subs}.")
    if reconciled.applied:
        parts.append(
            f"Supervisor applied a high-confidence reconciliation "
            f"({reconciled.disagreement.value} disagreement)."
        )
    else:
        parts.append("Supervisor fell back to the robust ensemble aggregate.")
    parts.append(
        f"Calibrated probability {calibrated.calibrated_probability:.3f} "
        f"(raw {calibrated.raw_probability:.3f})."
    )
    return " ".join(parts)


def build_forecast_input(
    *,
    question_id: str,
    as_of: datetime,
    evidence_set_id: str | None,
    calibrated: CalibratedForecast,
    uncertainty: Uncertainty,
    ensemble: EnsembleForecast,
    reconciled: ReconciledForecast,
    base_rate: BaseRateEstimate,
    decomposition: Decomposition,
    leakage: LeakageGateResult,
    rationale: str,
    model_provenance: dict[str, Any],
    repro_handle: dict[str, Any],
) -> ForecastInput:
    """Assemble the complete Forecast record input for the registry."""
    trace: dict[str, Any] = {
        "base_rate": dict(base_rate.provenance),
        "decomposition": dict(decomposition.provenance),
        "ensemble": dict(ensemble.provenance),
        "supervisor": dict(reconciled.provenance),
        "supervisor_trajectory": dict(reconciled.trajectory),
        "leakage": {
            "flagged": leakage.flagged,
            "verdicts": [v.model_dump(mode="json") for v in leakage.verdicts],
        },
    }
    calibration_metadata: dict[str, Any] = {
        **dict(calibrated.provenance),
        "near_decision_boundary": calibrated.near_decision_boundary,
        "quarantined": leakage.flagged,
    }
    return ForecastInput(
        question_id=question_id,
        as_of=as_of,
        probability=calibrated.calibrated_probability,
        rationale=rationale,
        evidence_set_id=evidence_set_id,
        model_provenance=model_provenance,
        trace=trace,
        calibration_metadata=calibration_metadata,
        uncertainty=asdict(uncertainty),
        repro_handle=repro_handle,
    )
