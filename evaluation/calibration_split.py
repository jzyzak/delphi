"""Calibration split management (C6.5).

Recalibration and extremization are learned ONLY on a dedicated calibration split
that is disjoint from the holdout and the live set (CLAUDE.md §2.5 — leaking the
holdout into calibration is a §2.2 violation in a lab coat). This module assigns
that disjoint split, fits an isotonic recalibrator (PAV) and an extremization
coefficient on it, and bundles them into a persistable artifact that the
forecast-chain calibrate stage (C4.5) consumes as a ``Recalibrator``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.forecast.calibration import DEFAULT_ALPHA, calibrate
from evaluation.scoring import ScoredRecord, log_score

__all__ = [
    "CalibrationArtifact",
    "IsotonicRecalibrator",
    "assign_calibration_split",
    "fit_calibration_artifact",
    "fit_extremization_coefficient",
    "fit_isotonic",
]

_EPS = 1e-6


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


def fit_isotonic(probabilities: Sequence[float], outcomes: Sequence[float]) -> IsotonicRecalibrator:
    """Fit a monotone recalibration map via Pool-Adjacent-Violators."""
    if len(probabilities) != len(outcomes):
        msg = "probabilities and outcomes must have equal length."
        raise ValueError(msg)
    if not probabilities:
        msg = "cannot fit a recalibrator on an empty set."
        raise ValueError(msg)
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


@dataclass(frozen=True)
class CalibrationArtifact:
    """Fitted recalibrator + extremization coefficient, persistable to JSON."""

    recalibrator: IsotonicRecalibrator
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {"recalibrator": self.recalibrator.to_dict(), "alpha": self.alpha}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationArtifact:
        return cls(
            recalibrator=IsotonicRecalibrator.from_dict(data["recalibrator"]),
            alpha=float(data["alpha"]),
        )


def fit_calibration_artifact(records: Sequence[ScoredRecord]) -> CalibrationArtifact:
    """Fit the recalibrator + extremization coefficient on calibration records."""
    if not records:
        msg = "cannot fit a calibration artifact on an empty split."
        raise ValueError(msg)
    probs = [r.probability for r in records]
    outcomes = [r.outcome for r in records]
    return CalibrationArtifact(
        recalibrator=fit_isotonic(probs, outcomes),
        alpha=fit_extremization_coefficient(probs, outcomes),
    )
