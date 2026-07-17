"""Calibrate + uncertainty stage (C4.5).

Maps the reconciled probability through (1) a learned recalibrator artifact fit
ONLY on the calibration split (identity until Phase 6 provides one — never fit
here, CLAUDE.md §2.5), then (2) the fixed-coefficient log-odds extremization from
the core. Uncertainty is quantified from ensemble spread + event uncertainty and
carries the sizing haircut.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.forecast.calibration import (
    DEFAULT_ALPHA,
    DEFAULT_BOUNDARY_MARGIN,
    DEFAULT_SPREAD_THRESHOLD,
    CalibratedForecast,
    apply_floor,
    calibrate,
    near_decision_boundary,
)
from core.forecast.supervisor import ReconciledForecast
from core.forecast.uncertainty import (
    Uncertainty,
    UncertaintyConfig,
    quantify_from_calibrated,
)

__all__ = [
    "IdentityRecalibrator",
    "Recalibrator",
    "calibrate_reconciled",
]


@runtime_checkable
class Recalibrator(Protocol):
    """A recalibration map fit on the disjoint calibration split (§2.5)."""

    def apply(self, probability: float) -> float:
        """Map a raw probability to a recalibrated one, both in [0, 1]."""
        ...

    @property
    def provenance(self) -> dict[str, Any]:
        """Identifying metadata for the fitted artifact (or identity)."""
        ...


class IdentityRecalibrator:
    """The no-op recalibrator used until Phase 6 fits a real artifact."""

    def apply(self, probability: float) -> float:
        return probability

    @property
    def provenance(self) -> dict[str, Any]:
        return {"recalibrator": "identity", "fitted": False}


def calibrate_reconciled(
    reconciled: ReconciledForecast,
    *,
    recalibrator: Recalibrator | None = None,
    alpha: float = DEFAULT_ALPHA,
    floor: float | None = None,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
    uncertainty_config: UncertaintyConfig | None = None,
) -> tuple[CalibratedForecast, Uncertainty]:
    """Recalibrate, extremize, and floor-clamp the reconciled probability.

    ``floor`` (fit on the calibration split, §2.5) clamps the final probability
    to ``[floor, 1 - floor]`` — the guard against confidently-wrong tails.
    """
    recal = recalibrator if recalibrator is not None else IdentityRecalibrator()
    raw_p = reconciled.probability
    recalibrated_p = recal.apply(raw_p)
    calibrated_p = apply_floor(calibrate(recalibrated_p, alpha=alpha), floor)
    spread = reconciled.uncertainty
    diagnostic = near_decision_boundary(
        recalibrated_p,
        spread,
        boundary_margin=boundary_margin,
        spread_threshold=spread_threshold,
    )
    calibrated = CalibratedForecast(
        calibrated_probability=calibrated_p,
        ensemble_uncertainty=spread,
        raw_probability=raw_p,
        near_decision_boundary=diagnostic,
        provenance={
            "calibration_method": "recalibrate_then_platt_logodds_extremization",
            "alpha": alpha,
            "floor": floor,
            "raw_probability": raw_p,
            "recalibrated_probability": recalibrated_p,
            "calibrated_probability": calibrated_p,
            "recalibrator": recal.provenance,
            "supervisor_applied": reconciled.applied,
        },
    )
    uncertainty = quantify_from_calibrated(calibrated, config=uncertainty_config)
    return calibrated, uncertainty
