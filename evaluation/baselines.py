"""Mandatory baselines (C6.3).

A proper score in isolation is meaningless (CLAUDE.md §2.3): every reported score
must be compared against the superforecaster median, the market/crowd consensus,
and a strong off-the-shelf LLM. Each baseline turns per-question inputs into a
probability that is then scored through the *same* harness path as the model.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "SUPERFORECASTER_MEDIAN",
    "MARKET_CONSENSUS",
    "STRONG_LLM",
    "Baseline",
    "clip_probability",
    "market_consensus",
    "strong_llm_baseline",
    "superforecaster_median",
]

SUPERFORECASTER_MEDIAN = "superforecaster_median"
MARKET_CONSENSUS = "market_consensus"
STRONG_LLM = "strong_llm"

_EPS = 1e-6


def clip_probability(p: float) -> float:
    """Clip a probability into the open unit interval for stable scoring."""
    return min(max(p, _EPS), 1.0 - _EPS)


def superforecaster_median(estimates: Sequence[float]) -> float:
    """Median of a panel of superforecaster probability estimates."""
    if not estimates:
        msg = "estimates must be non-empty."
        raise ValueError(msg)
    return clip_probability(statistics.median(estimates))


def market_consensus(price: float) -> float:
    """Market/crowd consensus price interpreted as a probability."""
    return clip_probability(price)


def strong_llm_baseline(probability: float) -> float:
    """A strong off-the-shelf LLM's single probability estimate."""
    return clip_probability(probability)


@dataclass(frozen=True)
class Baseline:
    """A named baseline: per-question predictions keyed by question id."""

    name: str
    predictions: Mapping[str, float]

    def predict(self, question_id: str) -> float | None:
        """Return this baseline's probability for a question, if it has one."""
        value = self.predictions.get(question_id)
        return None if value is None else clip_probability(value)
