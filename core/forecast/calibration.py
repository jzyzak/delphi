"""Platt / log-odds extremization calibration for ensemble aggregates.

Also home to :class:`FrozenCalibration` — the *apply-only* reader of the fitted
calibration artifact. Fitting lives in ``evaluation/calibration_split.py`` and
happens ONLY on the disjoint calibration split (CLAUDE.md §2.5); this module
never imports evaluation internals (§2.2 directionality) — it just loads and
applies the persisted JSON schema the two sides share.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.forecast.ensemble import EnsembleForecast

DEFAULT_ALPHA = math.sqrt(3)
DEFAULT_BOUNDARY_MARGIN = 0.05
DEFAULT_SPREAD_THRESHOLD = 0.15
_PROB_EPS = 1e-12
_RECAL_EPS = 1e-6  # matches the fitting side's clamp (evaluation/calibration_split.py)
CALIBRATION_SCHEMA_VERSION = 1


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


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation with flat extrapolation (np.interp semantics)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            # xs[i-1] == xs[i] is impossible here: x > xs[i-1] (no earlier
            # segment matched) and x <= xs[i] force xs[i-1] < xs[i].
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]  # pragma: no cover - unreachable: x < xs[-1] hits a segment


def _validate_floor(floor: float | None) -> None:
    if floor is None:
        return
    if not math.isfinite(floor) or not 0.0 <= floor < 0.5:
        msg = f"floor must be in [0, 0.5), got {floor!r}"
        raise ValueError(msg)


def apply_floor(p: float, floor: float | None) -> float:
    """Clamp a final probability to ``[floor, 1 - floor]`` (no-op when ``None``)."""
    _validate_probability(p)
    _validate_floor(floor)
    if floor is None:
        return p
    return min(max(p, floor), 1.0 - floor)


@dataclass(frozen=True)
class FrozenCalibration:
    """Apply-only fitted calibration: recalibrator + alpha + floor, from an artifact.

    Satisfies the forecast chain's ``Recalibrator`` protocol (``apply`` +
    ``provenance``); ``apply`` performs *recalibration only* — the chain applies
    the fitted ``alpha`` extremization and the ``floor`` clamp as its own stages,
    exactly as the fitting side composed them.
    """

    method: str  # "isotonic" | "platt"
    alpha: float
    floor: float | None = None
    x_knots: tuple[float, ...] = ()
    y_knots: tuple[float, ...] = ()
    a: float = 1.0
    b: float = 0.0
    n: int = 0
    artifact_hash: str = ""
    fitted_meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in ("isotonic", "platt"):
            msg = f"unknown recalibrator method {self.method!r}"
            raise ValueError(msg)
        if self.method == "isotonic" and (
            not self.x_knots or len(self.x_knots) != len(self.y_knots)
        ):
            msg = "isotonic calibration requires equal-length, non-empty knot arrays."
            raise ValueError(msg)
        _validate_alpha(self.alpha)
        _validate_floor(self.floor)

    def apply(self, probability: float) -> float:
        """Map a raw probability through the fitted recalibrator only."""
        _validate_probability(probability)
        if self.method == "isotonic":
            value = _interp(probability, self.x_knots, self.y_knots)
        else:
            clamped = min(max(probability, _RECAL_EPS), 1.0 - _RECAL_EPS)
            value = _sigmoid(self.a * math.log(clamped / (1.0 - clamped)) + self.b)
        return min(max(value, _RECAL_EPS), 1.0 - _RECAL_EPS)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "recalibrator": self.method,
            "fitted": True,
            "n": self.n,
            "alpha": self.alpha,
            "floor": self.floor,
            "artifact_hash": self.artifact_hash,
            **dict(self.fitted_meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, artifact_hash: str = "") -> FrozenCalibration:
        """Load from the artifact schema written by the evaluation fitting side."""
        schema_version = int(data.get("schema_version", CALIBRATION_SCHEMA_VERSION))
        if schema_version != CALIBRATION_SCHEMA_VERSION:
            msg = (
                f"unsupported calibration artifact schema_version {schema_version} "
                f"(expected {CALIBRATION_SCHEMA_VERSION})"
            )
            raise ValueError(msg)
        method = str(data.get("method", "isotonic"))
        if method not in ("isotonic", "platt"):
            msg = f"unknown recalibrator method {method!r}"
            raise ValueError(msg)
        recal = data["recalibrator"]
        floor_raw = data.get("floor")
        fitted_meta = dict(data.get("fitted", {}))
        # A fallback artifact is the labeled identity map from a starved fit;
        # the flag must survive into provenance so reports can surface it.
        fitted_meta["fallback"] = bool(data.get("fallback", False))
        common: dict[str, Any] = {
            "method": method,
            "alpha": float(data["alpha"]),
            "floor": None if floor_raw is None else float(floor_raw),
            "n": int(data.get("n", recal.get("n", 0))),
            "artifact_hash": artifact_hash,
            "fitted_meta": fitted_meta,
        }
        if method == "isotonic":
            return cls(
                x_knots=tuple(float(v) for v in recal["x"]),
                y_knots=tuple(float(v) for v in recal["y"]),
                **common,
            )
        return cls(a=float(recal["a"]), b=float(recal["b"]), **common)


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
