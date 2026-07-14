"""Platt / log-odds extremization calibration for ensemble aggregates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.forecast.ensemble import EnsembleForecast

DEFAULT_ALPHA = math.sqrt(3)
DEFAULT_BOUNDARY_MARGIN = 0.05
DEFAULT_SPREAD_THRESHOLD = 0.15
_PROB_EPS = 1e-12


def _clamp_probability(p: float) -> float:
    """Clamp ``p`` to the open unit interval for stable log-odds."""
    return min(max(p, _PROB_EPS), 1.0 - _PROB_EPS)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _validate_probability(p: float) -> None:
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        msg = f"probability must be finite and in [0, 1], got {p!r}"
        raise ValueError(msg)


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(alpha) or alpha <= 0.0:
        msg = f"alpha must be finite and positive, got {alpha!r}"
        raise ValueError(msg)


def calibrate(p: float, *, alpha: float = DEFAULT_ALPHA) -> float:
    """Extremize a probability toward 0/1: sigmoid(alpha * logit(p)).

    Default ``alpha = sqrt(3)`` is FIXED — never fit on outcome data. Amplifies
    whichever side of 0.5 ``p`` is on, so it helps only when ``p`` is already
    correct; it is downstream of evidence, not a fix for it. Apply ONCE, at the
    probability level; sizing (13) handles exposure separately.
    """
    _validate_probability(p)
    _validate_alpha(alpha)
    if p == 0.5:
        return 0.5
    clamped = _clamp_probability(p)
    logit = math.log(clamped / (1.0 - clamped))
    return _sigmoid(alpha * logit)


def near_decision_boundary(
    p: float,
    uncertainty: float,
    *,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
) -> bool:
    """Flag near-0.5 / low-confidence forecasts where extremization is risky.

    Extremization amplifies whichever side of 0.5 the base forecast is on. Near
    the decision boundary or with high ensemble spread, that amplification is
    especially dangerous.
    """
    _validate_probability(p)
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        msg = f"uncertainty must be finite and non-negative, got {uncertainty!r}"
        raise ValueError(msg)
    if not math.isfinite(boundary_margin) or boundary_margin < 0.0:
        msg = f"boundary_margin must be finite and non-negative, got {boundary_margin!r}"
        raise ValueError(msg)
    if not math.isfinite(spread_threshold) or spread_threshold < 0.0:
        msg = f"spread_threshold must be finite and non-negative, got {spread_threshold!r}"
        raise ValueError(msg)
    return abs(p - 0.5) < boundary_margin or uncertainty > spread_threshold


@dataclass(frozen=True)
class CalibratedForecast:
    """Calibrated ensemble output for downstream sizing (13).

    ``ensemble_uncertainty`` is the raw ensemble spread from prompt 18 — passed
    through unchanged, not re-extremized. Only ``calibrated_probability`` has
    the calibration transform applied, once.
    """

    calibrated_probability: float
    ensemble_uncertainty: float
    raw_probability: float
    near_decision_boundary: bool
    provenance: Mapping[str, Any]


def calibrate_ensemble(
    ensemble: EnsembleForecast,
    *,
    alpha: float = DEFAULT_ALPHA,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
) -> CalibratedForecast:
    """Apply fixed-coefficient calibration once to an ensemble aggregate.

    Calibrates ``ensemble.probability`` only. ``ensemble.uncertainty`` is passed
    through to sizing without modification.
    """
    raw_p = ensemble.probability
    calibrated_p = calibrate(raw_p, alpha=alpha)
    diagnostic = near_decision_boundary(
        raw_p,
        ensemble.uncertainty,
        boundary_margin=boundary_margin,
        spread_threshold=spread_threshold,
    )
    provenance: dict[str, Any] = {
        "calibration_method": "platt_logodds_extremization",
        "alpha": alpha,
        "boundary_margin": boundary_margin,
        "spread_threshold": spread_threshold,
        "raw_probability": raw_p,
        "calibrated_probability": calibrated_p,
        "ensemble_provenance": dict(ensemble.provenance),
    }
    return CalibratedForecast(
        calibrated_probability=calibrated_p,
        ensemble_uncertainty=ensemble.uncertainty,
        raw_probability=raw_p,
        near_decision_boundary=diagnostic,
        provenance=provenance,
    )
