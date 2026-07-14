"""Unit tests for registry record assembly (C4.7.b)."""

from __future__ import annotations

from datetime import UTC, datetime

from core.forecast.calibration import CalibratedForecast
from core.forecast.supervisor import Confidence, DisagreementKind, ReconciledForecast
from forecaster.record import build_rationale
from forecaster.stages.base_rate import BaseRateEstimate
from forecaster.stages.decompose import Decomposition, SubQuestion

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _calibrated(p: float = 0.7) -> CalibratedForecast:
    return CalibratedForecast(
        calibrated_probability=p,
        ensemble_uncertainty=0.05,
        raw_probability=0.6,
        near_decision_boundary=False,
        provenance={},
    )


def _reconciled(*, applied: bool) -> ReconciledForecast:
    return ReconciledForecast(
        probability=0.7,
        uncertainty=0.05,
        aggregate_probability=0.6,
        confidence=Confidence.HIGH if applied else Confidence.LOW,
        applied=applied,
        knowledge_time=AS_OF,
        disagreement=DisagreementKind.MULTIMODAL if applied else DisagreementKind.NONE,
    )


def test_rationale_full_with_supervisor_applied() -> None:
    base = BaseRateEstimate(prior=0.4, reference_class="rc", rationale="history")
    decomp = Decomposition(sub_questions=(SubQuestion(text="Will A?"),), rule="product")
    text = build_rationale(base, decomp, _reconciled(applied=True), _calibrated())
    assert "history" in text
    assert "Decomposed (product)" in text
    assert "Supervisor applied" in text


def test_rationale_minimal_with_fallback() -> None:
    base = BaseRateEstimate(prior=0.4, reference_class="rc")  # no rationale
    decomp = Decomposition()  # no sub-questions
    text = build_rationale(base, decomp, _reconciled(applied=False), _calibrated())
    assert "Decomposed" not in text
    assert "fell back" in text
