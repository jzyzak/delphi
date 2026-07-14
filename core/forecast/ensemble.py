"""Robust aggregation and spread computation for forecast ensembles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

from core.forecast.llm import ForecastDraw

Aggregator = Literal["median", "trimmed_mean"]
DEFAULT_TRIM_FRACTION = 0.1


def build_ensemble_config(
    *,
    n: int,
    aggregator: Aggregator,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> str:
    """Serialize ensemble parameters for content-addressed cache keys."""
    return f"n={n}|agg={aggregator}|trim={trim_fraction}|spread=std"


def aggregate(
    probabilities: Sequence[float],
    aggregator: Aggregator,
    *,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> float:
    """Aggregate N forecast probabilities with a robust statistic (no LLM combiner)."""
    if not probabilities:
        msg = "probabilities must be non-empty"
        raise ValueError(msg)
    arr = np.asarray(probabilities, dtype=np.float64)
    if aggregator == "median":
        return float(np.median(arr))
    if aggregator == "trimmed_mean":
        return _trimmed_mean(arr, trim_fraction)
    msg = f"unsupported aggregator: {aggregator!r}"
    raise ValueError(msg)


def _trimmed_mean(arr: np.ndarray, trim_fraction: float) -> float:
    """Trim symmetric tails then average the remainder."""
    if len(arr) < 3:
        return float(np.mean(arr))
    trim_fraction = max(0.0, min(trim_fraction, 0.49))
    n_trim = int(len(arr) * trim_fraction)
    if n_trim == 0:
        return float(np.mean(arr))
    sorted_arr = np.sort(arr)
    trimmed = sorted_arr[n_trim : len(arr) - n_trim]
    if len(trimmed) == 0:
        return float(np.median(arr))
    return float(np.mean(trimmed))


def ensemble_std(probabilities: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1) of raw pre-calibration probabilities.

    Exposes run-to-run disagreement for downstream sizing haircuts. The robust
    aggregator already absorbs outliers in the point estimate; std's job is to
    surface disagreement, not suppress it.
    """
    if len(probabilities) <= 1:
        return 0.0
    return float(np.std(probabilities, ddof=1))


@dataclass(frozen=True)
class EnsembleForecast:
    """Aggregated ensemble result with first-class uncertainty spread."""

    probability: float
    uncertainty: float
    n: int
    aggregator: Aggregator
    trim_fraction: float
    knowledge_time: datetime
    draws: tuple[ForecastDraw, ...]
    provenance: Mapping[str, Any]


def build_ensemble(
    draws: Sequence[ForecastDraw],
    *,
    aggregator: Aggregator,
    knowledge_time: datetime,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> EnsembleForecast:
    """Build an ensemble forecast from N structured draws."""
    if not draws:
        msg = "draws must be non-empty"
        raise ValueError(msg)
    probabilities = [d.probability for d in draws]
    agg_prob = aggregate(probabilities, aggregator, trim_fraction=trim_fraction)
    spread = ensemble_std(probabilities)
    provenance: dict[str, Any] = {
        "aggregation_method": aggregator,
        "trim_fraction": trim_fraction,
        "spread_metric": "std",
        "n_runs": len(draws),
        "run_provenance": [dict(d.provenance) for d in draws],
        "raw_probabilities": list(probabilities),
    }
    return EnsembleForecast(
        probability=agg_prob,
        uncertainty=spread,
        n=len(draws),
        aggregator=aggregator,
        trim_fraction=trim_fraction,
        knowledge_time=knowledge_time,
        draws=tuple(draws),
        provenance=provenance,
    )
