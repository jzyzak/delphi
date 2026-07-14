"""Direct tests for the envelope builder edge cases (C10.3)."""

from __future__ import annotations

from api.envelope import build_envelope
from core.registry.store import InMemoryRegistryStore
from forecaster.chain import ForecastResult


def _refused(*, refusal: object = None) -> ForecastResult:
    return ForecastResult(
        accepted=False,
        question_id=None,
        forecast_id=None,
        probability=None,
        calibrated=None,
        uncertainty=None,
        evidence=(),
        leakage=None,
        quarantined=False,
        rationale="",
        refusal=refusal,  # type: ignore[arg-type]
    )


def test_refusal_without_decision_has_empty_reason() -> None:
    envelope = build_envelope(_refused(), store=InMemoryRegistryStore())
    assert envelope.refused
    assert envelope.refusal_reason == ""
