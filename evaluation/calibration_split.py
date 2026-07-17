"""Calibration split management (C6.5).

Recalibration and extremization are learned ONLY on a dedicated calibration split
that is disjoint from the holdout and the live set (CLAUDE.md §2.5 — leaking the
holdout into calibration is a §2.2 violation in a lab coat). This module assigns
that disjoint split, fits a recalibrator (isotonic PAV or two-parameter Platt,
selected by K-fold cross-validation *within* the split) and an extremization
coefficient on it, and bundles them into a persistable artifact that the
forecast-chain calibrate stage (C4.5) consumes as a ``Recalibrator``.

The artifact JSON schema (``CalibrationArtifact.to_dict``) is the contract the
live chain reads through ``core.forecast.calibration.FrozenCalibration`` —
fitting lives here, application lives in core, and a parity test keeps the two
in lockstep.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.forecast.calibration import DEFAULT_ALPHA, calibrate
from evaluation.scoring import ScoredRecord, log_score

__all__ = [
    "DEFAULT_MIN_FIT_POINTS",
    "CalibrationArtifact",
    "IsotonicRecalibrator",
    "PlattRecalibrator",
    "assign_calibration_split",
    "fit_calibration_artifact",
    "fit_extremization_coefficient",
    "fit_isotonic",
    "fit_platt",
    "fit_probability_floor",
    "identity_fallback_artifact",
    "question_fingerprint",
    "select_recalibrator",
]

_EPS = 1e-6
_PLATT_RIDGE = 1e-6
_PLATT_MAX_ITER = 100
_PLATT_TOL = 1e-10
_PLATT_PARAM_BOUND = 20.0
ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_FLOOR_GRID = (0.0, 0.005, 0.01, 0.02, 0.05)


def assign_calibration_split(
    question_ids: Iterable[str],
    *,
    holdout_ids: Iterable[str] = (),
    live_ids: Iterable[str] = (),
    fraction: float = 1.0,
    seed: int = 0,
) -> frozenset[str]:
    """Assign a calibration split disjoint from the holdout and live sets.

    Reserved (holdout ∪ live) ids are removed first, so the returned set can
    never intersect them — the §2.5 disjointness invariant, enforced structurally.
    """
    if not 0.0 <= fraction <= 1.0:
        msg = "fraction must be in [0, 1]."
        raise ValueError(msg)
    reserved = set(holdout_ids) | set(live_ids)
    pool = sorted(set(question_ids) - reserved)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)  # type: ignore[arg-type]
    k = round(fraction * len(pool))
    calibration = frozenset(pool[:k])
    if calibration & reserved:  # pragma: no cover - impossible by construction
        msg = "calibration split intersects holdout/live (invariant violation)."
        raise RuntimeError(msg)
    return calibration


def _validate_pairs(probabilities: Sequence[float], outcomes: Sequence[float]) -> None:
    if len(probabilities) != len(outcomes):
        msg = "probabilities and outcomes must have equal length."
        raise ValueError(msg)
    if not probabilities:
        msg = "cannot fit a recalibrator on an empty set."
        raise ValueError(msg)


def fit_isotonic(probabilities: Sequence[float], outcomes: Sequence[float]) -> IsotonicRecalibrator:
    """Fit a monotone recalibration map via Pool-Adjacent-Violators."""
    _validate_pairs(probabilities, outcomes)
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    xs = [float(probabilities[i]) for i in order]
    ys = [float(outcomes[i]) for i in order]

    # PAV: merge adjacent blocks that violate monotonicity (value_prev > value_cur).
    blocks: list[list[float]] = []  # each block: [sum, count]
    for y in ys:
        blocks.append([y, 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (
            blocks[-1][0] / blocks[-1][1]
        ):
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2])

    fitted: list[float] = []
    for s, c in blocks:
        fitted.extend([s / c] * int(c))
    return IsotonicRecalibrator(x_thresholds=tuple(xs), y_values=tuple(fitted), n=len(xs))


@dataclass(frozen=True)
class IsotonicRecalibrator:
    """Monotone (isotonic) recalibration map; interpolates between fitted knots."""

    x_thresholds: tuple[float, ...]
    y_values: tuple[float, ...]
    n: int

    method = "isotonic"

    def apply(self, probability: float) -> float:
        value = float(np.interp(probability, self.x_thresholds, self.y_values))
        return min(max(value, _EPS), 1.0 - _EPS)

    @property
    def provenance(self) -> dict[str, Any]:
        return {"recalibrator": "isotonic_pav", "fitted": True, "n": self.n}

    def to_dict(self) -> dict[str, Any]:
        return {"x": list(self.x_thresholds), "y": list(self.y_values), "n": self.n}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsotonicRecalibrator:
        return cls(
            x_thresholds=tuple(float(v) for v in data["x"]),
            y_values=tuple(float(v) for v in data["y"]),
            n=int(data["n"]),
        )


def _clamp_probability(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def _logit(p: float) -> float:
    clamped = _clamp_probability(p)
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def fit_platt(probabilities: Sequence[float], outcomes: Sequence[float]) -> PlattRecalibrator:
    """Fit a two-parameter Platt map ``sigmoid(a * logit(p) + b)`` by log loss.

    Newton-Raphson with a tiny ridge for stability under separation; parameters
    are bounded to keep the map numerically sane on degenerate splits. A fitted
    slope ``a < 0`` would *reverse* forecasts — an overfit artifact on small
    splits, not a recalibration — so it deterministically falls back to the
    constant base-rate map (``a = 0``, ``b = logit(mean outcome)``).
    """
    _validate_pairs(probabilities, outcomes)
    z = np.array([_logit(float(p)) for p in probabilities])
    y = np.array([float(o) for o in outcomes])
    a, b = 1.0, 0.0
    for _ in range(_PLATT_MAX_ITER):
        mu = 1.0 / (1.0 + np.exp(-np.clip(a * z + b, -700.0, 700.0)))
        grad_a = float(np.mean((mu - y) * z)) + _PLATT_RIDGE * a
        grad_b = float(np.mean(mu - y)) + _PLATT_RIDGE * b
        w = mu * (1.0 - mu)
        h_aa = float(np.mean(w * z * z)) + _PLATT_RIDGE
        h_ab = float(np.mean(w * z))
        h_bb = float(np.mean(w)) + _PLATT_RIDGE
        det = h_aa * h_bb - h_ab * h_ab
        # Cauchy-Schwarz + the ridge keep det > 0 on finite inputs; this guard
        # only fires on NaN propagation from pathological floats.
        if det <= 0.0 or not math.isfinite(det):  # pragma: no cover - defensive
            break
        step_a = (h_bb * grad_a - h_ab * grad_b) / det
        step_b = (h_aa * grad_b - h_ab * grad_a) / det
        a = min(max(a - step_a, -_PLATT_PARAM_BOUND), _PLATT_PARAM_BOUND)
        b = min(max(b - step_b, -_PLATT_PARAM_BOUND), _PLATT_PARAM_BOUND)
        if max(abs(step_a), abs(step_b)) < _PLATT_TOL:
            break
    if a < 0.0:
        a = 0.0
        b = _logit(float(np.mean(y)))
    return PlattRecalibrator(a=float(a), b=float(b), n=len(probabilities))


@dataclass(frozen=True)
class PlattRecalibrator:
    """Two-parameter logistic recalibration map ``sigmoid(a * logit(p) + b)``."""

    a: float
    b: float
    n: int

    method = "platt"

    def apply(self, probability: float) -> float:
        value = _sigmoid(self.a * _logit(probability) + self.b)
        return min(max(value, _EPS), 1.0 - _EPS)

    @property
    def provenance(self) -> dict[str, Any]:
        return {"recalibrator": "platt", "fitted": True, "n": self.n, "a": self.a, "b": self.b}

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "n": self.n}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlattRecalibrator:
        return cls(a=float(data["a"]), b=float(data["b"]), n=int(data["n"]))


Fitted = IsotonicRecalibrator | PlattRecalibrator

# Below this, isotonic memorizes the split (a 9-point pilot fit turned raw
# Brier 0.237 into 0.323): Platt (2 params) is taken outright, and isotonic
# must earn its way in on >= 30 points via CV.
_MIN_CV_POINTS = 30
# Below this, no fit is trustworthy at all: fitting returns the labeled
# identity fallback so a starved fit can never degrade the score.
DEFAULT_MIN_FIT_POINTS = 10


def question_fingerprint(value: str) -> str:
    """Stable fingerprint for §2.5 disjointness checks (ids or normalized text)."""
    normalized = " ".join(value.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cv_log_loss(
    fit_fn: Any,
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    folds: int,
    seed: int,
) -> float:
    """Mean out-of-fold log loss of ``fit_fn`` under K-fold CV within the split."""
    n = len(probabilities)
    k = min(folds, n)
    rng = np.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(n)]
    losses: list[float] = []
    for fold in range(k):
        test_idx = set(perm[fold::k])
        train_idx = [i for i in range(n) if i not in test_idx]
        if not train_idx:  # pragma: no cover - k <= n guarantees a train side
            continue
        recal = fit_fn([probabilities[i] for i in train_idx], [outcomes[i] for i in train_idx])
        losses.extend(
            log_score(recal.apply(probabilities[i]), outcomes[i]) for i in sorted(test_idx)
        )
    return float(np.mean(losses))


def select_recalibrator(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    cv_folds: int = 5,
    seed: int = 0,
) -> Fitted:
    """Select isotonic vs Platt by K-fold CV *within* the calibration split.

    Small splits (< 8 points) skip CV and take Platt — two parameters, where raw
    isotonic would memorize the points. Ties prefer Platt for the same reason.
    """
    _validate_pairs(probabilities, outcomes)
    if len(probabilities) < _MIN_CV_POINTS or cv_folds < 2:
        return fit_platt(probabilities, outcomes)
    platt_loss = _cv_log_loss(fit_platt, probabilities, outcomes, folds=cv_folds, seed=seed)
    isotonic_loss = _cv_log_loss(fit_isotonic, probabilities, outcomes, folds=cv_folds, seed=seed)
    if isotonic_loss < platt_loss:
        return fit_isotonic(probabilities, outcomes)
    return fit_platt(probabilities, outcomes)


def fit_extremization_coefficient(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
) -> float:
    """Grid-search the extremization ``alpha`` minimizing mean log loss on the split."""
    if not probabilities:
        msg = "cannot fit an extremization coefficient on an empty set."
        raise ValueError(msg)
    if not grid:
        msg = "grid must be non-empty."
        raise ValueError(msg)
    best_alpha = DEFAULT_ALPHA
    best_loss = float("inf")
    for alpha in grid:
        loss = sum(
            log_score(calibrate(p, alpha=alpha), o)
            for p, o in zip(probabilities, outcomes, strict=True)
        ) / len(probabilities)
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
    return best_alpha


def fit_probability_floor(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    grid: Sequence[float] = DEFAULT_FLOOR_GRID,
) -> float | None:
    """Grid-fit a symmetric probability floor minimizing mean log loss.

    The floor clamps final probabilities to ``[floor, 1 - floor]`` — the defense
    against confidently-wrong tails (log loss punishes a wrong ~0.001 hardest,
    which is exactly the failure it exists to prevent). ``probabilities`` must be
    the *fully composed* outputs (recalibrated + extremized), so fit-time
    composition matches apply-time composition. Returns ``None`` when no positive
    floor beats the unclamped map.
    """
    _validate_pairs(probabilities, outcomes)
    if not grid:
        msg = "grid must be non-empty."
        raise ValueError(msg)
    best_floor = 0.0
    best_loss = float("inf")
    for floor in grid:
        if not 0.0 <= floor < 0.5:
            msg = f"floor grid values must be in [0, 0.5), got {floor!r}"
            raise ValueError(msg)
        loss = sum(
            log_score(min(max(p, floor), 1.0 - floor), o)
            for p, o in zip(probabilities, outcomes, strict=True)
        ) / len(probabilities)
        if loss < best_loss:
            best_loss = loss
            best_floor = floor
    return best_floor if best_floor > 0.0 else None


@dataclass(frozen=True)
class CalibrationArtifact:
    """Fitted recalibrator + extremization coefficient + floor, persistable to JSON."""

    recalibrator: Fitted
    alpha: float
    floor: float | None = None
    # True when the fit set was too small to trust and the artifact is the
    # labeled identity map (raw pass-through + DEFAULT_ALPHA, no floor).
    fallback: bool = False

    def apply(self, probability: float) -> float:
        """The full composed map: recalibrate → extremize → clamp to the floor."""
        calibrated = calibrate(self.recalibrator.apply(probability), alpha=self.alpha)
        if self.floor is not None:
            calibrated = min(max(calibrated, self.floor), 1.0 - self.floor)
        return calibrated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "method": self.recalibrator.method,
            "recalibrator": self.recalibrator.to_dict(),
            "alpha": self.alpha,
            "floor": self.floor,
            "n": self.recalibrator.n,
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationArtifact:
        method = data.get("method", "isotonic")  # pre-schema_version dicts were isotonic-only
        recalibrator: Fitted
        if method == "isotonic":
            recalibrator = IsotonicRecalibrator.from_dict(data["recalibrator"])
        elif method == "platt":
            recalibrator = PlattRecalibrator.from_dict(data["recalibrator"])
        else:
            msg = f"unknown recalibrator method {method!r}"
            raise ValueError(msg)
        floor_raw = data.get("floor")
        return cls(
            recalibrator=recalibrator,
            alpha=float(data["alpha"]),
            floor=None if floor_raw is None else float(floor_raw),
            fallback=bool(data.get("fallback", False)),
        )


def identity_fallback_artifact(n: int) -> CalibrationArtifact:
    """The labeled do-no-harm artifact for starved fits.

    ``PlattRecalibrator(a=1, b=0)`` is exactly the identity map
    (``sigmoid(1·logit(p) + 0) = p``), so the composed map equals the
    uncalibrated default path (``calibrate(p, DEFAULT_ALPHA)``) — a starved
    fit is structurally incapable of degrading the score.
    """
    return CalibrationArtifact(
        recalibrator=PlattRecalibrator(a=1.0, b=0.0, n=n),
        alpha=DEFAULT_ALPHA,
        floor=None,
        fallback=True,
    )


def fit_calibration_artifact(
    records: Sequence[ScoredRecord],
    *,
    method: str = "auto",
    cv_folds: int = 5,
    seed: int = 0,
    floor_grid: Sequence[float] = DEFAULT_FLOOR_GRID,
    min_fit_points: int = DEFAULT_MIN_FIT_POINTS,
) -> CalibrationArtifact:
    """Fit recalibrator + extremization coefficient + floor on calibration records.

    Fit-time composition matches apply-time composition: the recalibrator is fit
    (or CV-selected) first, ``alpha`` is fit on the *recalibrated* probabilities,
    and the floor is fit on the fully composed (recalibrated + extremized) map.
    Below ``min_fit_points`` records, no fit is trustworthy: the labeled
    identity fallback is returned instead (never a memorized map).
    """
    if method not in ("auto", "isotonic", "platt"):
        msg = f"unknown method {method!r}; choose 'auto', 'isotonic', or 'platt'."
        raise ValueError(msg)
    if not records:
        msg = "cannot fit a calibration artifact on an empty split."
        raise ValueError(msg)
    if len(records) < min_fit_points:
        return identity_fallback_artifact(len(records))
    probs = [r.probability for r in records]
    outcomes = [r.outcome for r in records]
    if method == "auto":
        recalibrator = select_recalibrator(probs, outcomes, cv_folds=cv_folds, seed=seed)
    elif method == "isotonic":
        recalibrator = fit_isotonic(probs, outcomes)
    else:
        recalibrator = fit_platt(probs, outcomes)
    recalibrated = [recalibrator.apply(p) for p in probs]
    alpha = fit_extremization_coefficient(recalibrated, outcomes)
    composed = [calibrate(p, alpha=alpha) for p in recalibrated]
    floor = fit_probability_floor(composed, outcomes, grid=floor_grid)
    return CalibrationArtifact(recalibrator=recalibrator, alpha=alpha, floor=floor)
