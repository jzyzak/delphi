"""Decompose forecast uncertainty into event vs LLM-output sources."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from core.forecast.calibration import CalibratedForecast
from core.forecast.ensemble import EnsembleForecast

# Chosen so spread ≈ DEFAULT_SPREAD_THRESHOLD (0.15) yields haircut ≈ 0.57.
DEFAULT_HAIRCUT_SENSITIVITY = 5.0


class UncertaintyConfig(BaseModel):
    """Fixed uncertainty-quantification parameters — not fitted on outcomes."""

    model_config = ConfigDict(frozen=True)

    haircut_sensitivity: float = DEFAULT_HAIRCUT_SENSITIVITY

    @field_validator("haircut_sensitivity")
    @classmethod
    def _positive(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            msg = f"haircut_sensitivity must be finite and positive, got {v!r}"
            raise ValueError(msg)
        return v


def _validate_probability(p: float) -> None:
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        msg = f"probability must be finite and in [0, 1], got {p!r}"
        raise ValueError(msg)


def _validate_spread(spread: float) -> None:
    if not math.isfinite(spread) or spread < 0.0:
        msg = f"ensemble_spread must be finite and non-negative, got {spread!r}"
        raise ValueError(msg)


def event_uncertainty(p: float) -> float:
    """Bernoulli standard deviation sqrt(p * (1 - p)).

    Peaks at p = 0.5; approaches 0 as p approaches 0 or 1.
    """
    _validate_probability(p)
    return math.sqrt(p * (1.0 - p))


def llm_output_uncertainty(ensemble_spread: float) -> float:
    """Run-to-run instability from the ensemble spread (prompt 18)."""
    _validate_spread(ensemble_spread)
    return float(ensemble_spread)


def combine_uncertainty(event: float, llm: float) -> float:
    """Combine distinct sources via variance addition."""
    if not math.isfinite(event) or event < 0.0:
        msg = f"event_uncertainty must be finite and non-negative, got {event!r}"
        raise ValueError(msg)
    if not math.isfinite(llm) or llm < 0.0:
        msg = f"llm_output_uncertainty must be finite and non-negative, got {llm!r}"
        raise ValueError(msg)
    return math.sqrt(event * event + llm * llm)


def stability_haircut(llm: float, *, config: UncertaintyConfig | None = None) -> float:
    """Sizing multiplier in (0, 1] from LLM-output instability only.

    Higher instability → smaller multiplier. Independent of event-uncertainty.
    """
    cfg = config or UncertaintyConfig()
    _validate_spread(llm)
    return 1.0 / (1.0 + cfg.haircut_sensitivity * llm)


@dataclass(frozen=True)
class Uncertainty:
    """Structured uncertainty record attached to each forecast.

    ``llm_output_uncertainty`` is the sizing haircut signal consumed by prompt 13.
    The two sources are kept distinct; ``combined`` is derived, not a replacement.
    """

    event_uncertainty: float
    llm_output_uncertainty: float
    combined: float
    stability_haircut: float
    provenance: Mapping[str, Any]


def quantify_uncertainty(
    probability: float,
    ensemble_spread: float,
    *,
    config: UncertaintyConfig | None = None,
) -> Uncertainty:
    """Return decomposed uncertainty for a forecast probability and ensemble spread.

    The two sources are kept distinct. ``llm_output_uncertainty`` is the sizing
    haircut consumed by 13: a model-unstable forecast is sized smaller regardless
    of how confident it is about the event.
    """
    cfg = config or UncertaintyConfig()
    event = event_uncertainty(probability)
    llm = llm_output_uncertainty(ensemble_spread)
    combined = combine_uncertainty(event, llm)
    haircut = stability_haircut(llm, config=cfg)
    provenance: dict[str, Any] = {
        "probability": probability,
        "ensemble_spread": ensemble_spread,
        "event_metric": "bernoulli_std",
        "llm_metric": "ensemble_spread_passthrough",
        "combination": "variance_addition",
        "haircut_form": "reciprocal",
        "haircut_sensitivity": cfg.haircut_sensitivity,
    }
    return Uncertainty(
        event_uncertainty=event,
        llm_output_uncertainty=llm,
        combined=combined,
        stability_haircut=haircut,
        provenance=provenance,
    )


def quantify_from_ensemble(
    ensemble: EnsembleForecast,
    *,
    config: UncertaintyConfig | None = None,
) -> Uncertainty:
    """Quantify uncertainty from an ``EnsembleForecast`` aggregate + spread."""
    return quantify_uncertainty(
        ensemble.probability,
        ensemble.uncertainty,
        config=config,
    )


def quantify_from_calibrated(
    calibrated: CalibratedForecast,
    *,
    config: UncertaintyConfig | None = None,
) -> Uncertainty:
    """Quantify uncertainty from a calibrated ensemble (19) output."""
    return quantify_uncertainty(
        calibrated.calibrated_probability,
        calibrated.ensemble_uncertainty,
        config=config,
    )


def apply_stability_haircut(exposure: float, uncertainty: Uncertainty) -> float:
    """Apply the LLM-output stability haircut to a sizing input (prompt 13)."""
    if not math.isfinite(exposure):
        msg = f"exposure must be finite, got {exposure!r}"
        raise ValueError(msg)
    return exposure * uncertainty.stability_haircut
