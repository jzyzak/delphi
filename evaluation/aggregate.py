"""Bootstrap CIs + per-domain aggregation + baseline deltas (C6.4).

A score without an interval is a point estimate pretending to be a fact. This
module resamples at the *question* level (the unit of independence) to put a
confidence interval on every mean score, groups scores per domain (§2.3), and
reports the model's delta vs each baseline through the same path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evaluation.baselines import Baseline
from evaluation.scoring import ScoredRecord, Scorer

__all__ = [
    "ScoreSummary",
    "baseline_delta",
    "bootstrap_ci",
    "per_domain_summary",
    "summarize_scores",
]


@dataclass(frozen=True)
class ScoreSummary:
    """A mean score with a question-level bootstrap confidence interval."""

    scorer: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (question-level resample)."""
    if not values:
        msg = "values must be non-empty."
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = "alpha must be in (0, 1)."
        raise ValueError(msg)
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _score_values(scorer: Scorer, records: Sequence[ScoredRecord]) -> list[float]:
    return [scorer.score(r.probability, r.outcome) for r in records]


def summarize_scores(
    scorer: Scorer,
    records: Sequence[ScoredRecord],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> ScoreSummary:
    """Mean score + bootstrap CI over ``records``."""
    if not records:
        msg = "records must be non-empty."
        raise ValueError(msg)
    values = _score_values(scorer, records)
    mean = float(np.mean(values))
    lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return ScoreSummary(scorer=scorer.name, mean=mean, ci_low=lo, ci_high=hi, n=len(records))


def per_domain_summary(
    scorer: Scorer,
    records: Sequence[ScoredRecord],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, ScoreSummary]:
    """Score summaries grouped by domain, in sorted domain order."""
    by_domain: dict[str, list[ScoredRecord]] = {}
    for record in records:
        by_domain.setdefault(record.domain, []).append(record)
    return {
        domain: summarize_scores(scorer, by_domain[domain], n_boot=n_boot, seed=seed)
        for domain in sorted(by_domain)
    }


def baseline_delta(
    scorer: Scorer, records: Sequence[ScoredRecord], baseline: Baseline
) -> float | None:
    """Mean(model score) - mean(baseline score) over questions the baseline covers.

    Negative means the model beats the baseline (lower proper score is better).
    Returns ``None`` if the baseline covers none of the records.
    """
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    for record in records:
        predicted = baseline.predict(record.question_id)
        if predicted is None:
            continue
        model_scores.append(scorer.score(record.probability, record.outcome))
        baseline_scores.append(scorer.score(predicted, record.outcome))
    if not model_scores:
        return None
    return float(np.mean(model_scores) - np.mean(baseline_scores))
