"""Robust aggregation and spread computation for forecast ensembles.

Log-odds pooling (CLAUDE.md §3.5 — combine in log-odds space) is the default
combination for the forecast chain: probabilities are clamped, mapped to
logits, averaged (optionally with symmetric trimming *in logit space*), and
mapped back. The probability-space ``median``/``trimmed_mean`` aggregators
remain for comparison and cache compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

from core.forecast.llm import ForecastDraw

Aggregator = Literal["median", "trimmed_mean", "log_odds_mean", "log_odds_trimmed_mean"]
DEFAULT_TRIM_FRACTION = 0.1
# Pooling-specific clamp: a draw at exactly 0/1 must not dominate the mean
# logit, so this is deliberately much looser than the calibration-side 1e-12.
DEFAULT_POOL_EPS = 1e-3
_LOG_ODDS_AGGREGATORS = ("log_odds_mean", "log_odds_trimmed_mean")


def build_ensemble_config(
    *,
    n: int,
    aggregator: Aggregator,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    pool_eps: float = DEFAULT_POOL_EPS,
) -> str:
    """Serialize ensemble parameters for content-addressed cache keys."""
    config = f"n={n}|agg={aggregator}|trim={trim_fraction}|spread=std"
    if aggregator in _LOG_ODDS_AGGREGATORS:
        config += f"|pool_eps={pool_eps}"
    return config


def _validate_pool_eps(pool_eps: float) -> None:
    if not 0.0 < pool_eps < 0.5:
        msg = f"pool_eps must be in (0, 0.5), got {pool_eps!r}"
        raise ValueError(msg)


def _log_odds_pool(arr: np.ndarray, *, trim_fraction: float, pool_eps: float) -> float:
    """Mean (optionally trimmed, in logit space) of logits, mapped back via sigmoid."""
    clamped = np.clip(arr, pool_eps, 1.0 - pool_eps)
    logits = np.log(clamped / (1.0 - clamped))
    mean_logit = _trimmed_mean(logits, trim_fraction)  # trim_fraction=0 -> plain mean
    return float(1.0 / (1.0 + np.exp(-mean_logit)))


def aggregate(
    probabilities: Sequence[float],
    aggregator: Aggregator,
    *,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
    pool_eps: float = DEFAULT_POOL_EPS,
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
    if aggregator in _LOG_ODDS_AGGREGATORS:
        _validate_pool_eps(pool_eps)
        trim = trim_fraction if aggregator == "log_odds_trimmed_mean" else 0.0
        return _log_odds_pool(arr, trim_fraction=trim, pool_eps=pool_eps)
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
    if len(trimmed) == 0:  # pragma: no cover - unreachable: trim <= 0.49 keeps >= 1
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
    pool_eps: float = DEFAULT_POOL_EPS,
) -> EnsembleForecast:
    """Build an ensemble forecast from N structured draws."""
    if not draws:
        msg = "draws must be non-empty"
        raise ValueError(msg)
    probabilities = [d.probability for d in draws]
    agg_prob = aggregate(probabilities, aggregator, trim_fraction=trim_fraction, pool_eps=pool_eps)
    spread = ensemble_std(probabilities)
    provenance: dict[str, Any] = {
        "aggregation_method": aggregator,
        "trim_fraction": trim_fraction,
        "spread_metric": "std",
        "n_runs": len(draws),
        "run_provenance": [dict(d.provenance) for d in draws],
        "raw_probabilities": list(probabilities),
    }
    if aggregator in _LOG_ODDS_AGGREGATORS:
        provenance["pool_eps"] = pool_eps
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
