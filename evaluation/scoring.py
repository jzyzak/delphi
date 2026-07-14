"""Proper scoring rules (C6.1).

Proper scores only — never accuracy (CLAUDE.md §2.3). A proper score is minimized
in expectation by reporting one's true belief, so it cannot be gamed by
overconfidence. Binary forecasts are scored with Brier and log loss; predictive
distributions with CRPS (approximated from a quantile set via averaged pinball
loss). All scores here are *lower-is-better* (negatively oriented).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "BrierScorer",
    "CRPSScorer",
    "LogScorer",
    "ScoredRecord",
    "Scorer",
    "brier_score",
    "crps_from_quantiles",
    "log_score",
    "mean_score",
]

_LOG_EPS = 1e-15


def _validate_probability(p: float) -> None:
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        msg = f"probability must be finite and in [0, 1], got {p!r}"
        raise ValueError(msg)


def _validate_outcome(o: float) -> None:
    if o not in (0.0, 1.0):
        msg = f"binary outcome must be 0.0 or 1.0, got {o!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ScoredRecord:
    """A resolved binary forecast ready for scoring, with baseline predictions."""

    question_id: str
    domain: str
    probability: float
    outcome: float
    baselines: Mapping[str, float] = field(default_factory=dict)


def brier_score(probability: float, outcome: float) -> float:
    """Squared error ``(p - o)^2`` for a binary forecast (lower is better)."""
    _validate_probability(probability)
    _validate_outcome(outcome)
    return (probability - outcome) ** 2


def log_score(probability: float, outcome: float) -> float:
    """Negative log-likelihood of the outcome (lower is better), clamped for stability."""
    _validate_probability(probability)
    _validate_outcome(outcome)
    p = min(max(probability, _LOG_EPS), 1.0 - _LOG_EPS)
    return -(outcome * math.log(p) + (1.0 - outcome) * math.log(1.0 - p))


def crps_from_quantiles(quantiles: Sequence[tuple[float, float]], outcome: float) -> float:
    """CRPS approximated as twice the mean pinball loss over a quantile set.

    ``quantiles`` is a sequence of ``(level, value)`` pairs with levels in (0, 1).
    The pinball (quantile) loss is a proper scoring rule for each quantile; its
    average over a dense, symmetric grid approximates the CRPS.
    """
    if not quantiles:
        msg = "quantiles must be non-empty."
        raise ValueError(msg)
    total = 0.0
    for level, value in quantiles:
        if not 0.0 < level < 1.0:
            msg = f"quantile level must be in (0, 1), got {level!r}"
            raise ValueError(msg)
        delta = outcome - value
        loss = level * delta if delta >= 0 else (level - 1.0) * delta
        total += loss
    return 2.0 * total / len(quantiles)


@runtime_checkable
class Scorer(Protocol):
    """A proper scoring rule (negatively oriented: lower is better)."""

    @property
    def name(self) -> str: ...

    def score(self, prediction: float, outcome: float) -> float:
        """Score one prediction against its realized outcome."""
        ...


class BrierScorer:
    name = "brier"

    def score(self, prediction: float, outcome: float) -> float:
        return brier_score(prediction, outcome)


class LogScorer:
    name = "log"

    def score(self, prediction: float, outcome: float) -> float:
        return log_score(prediction, outcome)


class CRPSScorer:
    """Scores a predictive distribution given as a quantile set."""

    name = "crps"

    def score(self, prediction: Sequence[tuple[float, float]], outcome: float) -> float:
        return crps_from_quantiles(prediction, outcome)


def mean_score(scorer: Scorer, predictions: Sequence[float], outcomes: Sequence[float]) -> float:
    """Mean of ``scorer`` over paired predictions and outcomes."""
    if len(predictions) != len(outcomes):
        msg = "predictions and outcomes must have equal length."
        raise ValueError(msg)
    if not predictions:
        msg = "cannot score an empty set."
        raise ValueError(msg)
    return sum(scorer.score(p, o) for p, o in zip(predictions, outcomes, strict=True)) / len(
        predictions
    )
