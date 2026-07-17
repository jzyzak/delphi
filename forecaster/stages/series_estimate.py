"""Deterministic series-threshold estimator (quantitative reference class).

For questions of the shape "will this numeric series be higher on the
resolution date than at the forecast time" (ForecastBench fred / yfinance /
dbnomics questions), the honest base rate is arithmetic, not judgment: the
empirical frequency of positive h-day moves in the series' own history. The
LLM pipeline reads text and cannot do this reliably — measured directly as
the FRED/yfinance domain gap. This stage computes that frequency from as-of
history (season-matched when the data supports it, Laplace-smoothed) and
injects it as a high-salience evidence item, exactly like the market freeze
value. Leakage-safe by construction: the history provider returns only
observations dated at or before the ceiling, and the estimate carries
``knowledge_time == as_of``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

import structlog

from core.forecast.search import Evidence
from core.pit.models import ensure_utc
from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

__all__ = [
    "SERIES_ESTIMATOR_SOURCE",
    "SeriesEstimate",
    "SeriesEvidenceEstimator",
    "SeriesHistorySource",
    "estimate_direction_probability",
]

_LOG = structlog.get_logger(__name__)

SERIES_ESTIMATOR_SOURCE = "series_estimator"
DEFAULT_MIN_SAMPLES = 30
DEFAULT_SEASONAL_WINDOW_DAYS = 45


@runtime_checkable
class SeriesHistorySource(Protocol):
    """Where the estimator gets as-of series history (see sources/series.py)."""

    def history(
        self, benchmark_question_id: str, *, as_of: datetime
    ) -> tuple[tuple[date, float], ...]: ...


@dataclass(frozen=True)
class SeriesEstimate:
    """A deterministic direction-probability estimate from series history."""

    probability: float
    n_samples: int
    horizon_days: int
    season_matched: bool


def _match_tolerance(horizon_days: int) -> int:
    """Nearest-observation tolerance: wider for longer horizons, floor of 2 days."""
    return max(2, horizon_days // 7)


def _doy_distance(a: date, b: date) -> int:
    """Circular day-of-year distance (Dec 28 is 4 days from Jan 1)."""
    delta = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(delta, 365 - delta)


def estimate_direction_probability(
    history: tuple[tuple[date, float], ...],
    *,
    as_of: datetime,
    resolution_date: datetime,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seasonal_window_days: int = DEFAULT_SEASONAL_WINDOW_DAYS,
) -> SeriesEstimate | None:
    """Empirical P(value at resolution > value at as-of) from h-day moves.

    Every historical observation that has a matching observation ``h`` days
    later (within a small tolerance) contributes one sample: did the series
    rise over that window? If enough samples start near the same time of year
    as ``as_of``, only those are used (seasonal series — temperature — trend
    by calendar); otherwise all samples count. Laplace smoothing keeps the
    estimate off the 0/1 poles. Returns ``None`` when the horizon or history
    is unusable — the estimator must stay silent rather than guess.
    """
    as_of = ensure_utc(as_of)
    resolution_date = ensure_utc(resolution_date)
    horizon = (resolution_date.date() - as_of.date()).days
    if horizon < 1 or len(history) < 2:
        return None

    observations = sorted(history)
    dates = [observed for observed, _ in observations]
    tolerance = _match_tolerance(horizon)

    samples: list[tuple[date, bool]] = []  # (window start, series rose)
    j = 0
    for i, (start, start_value) in enumerate(observations):
        target = start.toordinal() + horizon
        # Advance a single cursor (both lists are sorted) to the first
        # observation at/after the target, then compare with its predecessor.
        j = max(j, i + 1)
        while j < len(observations) and dates[j].toordinal() < target:
            j += 1
        best: tuple[int, float] | None = None
        for k in (j - 1, j):
            if k <= i or k >= len(observations):
                continue
            distance = abs(dates[k].toordinal() - target)
            if distance <= tolerance and (best is None or distance < best[0]):
                best = (distance, observations[k][1])
        if best is not None:
            samples.append((start, best[1] > start_value))

    if len(samples) < min_samples:
        return None

    seasonal = [
        rose
        for start, rose in samples
        if _doy_distance(start, as_of.date()) <= seasonal_window_days
    ]
    season_matched = len(seasonal) >= min_samples
    used = seasonal if season_matched else [rose for _, rose in samples]

    positives = sum(1 for rose in used if rose)
    probability = (positives + 1) / (len(used) + 2)
    return SeriesEstimate(
        probability=probability,
        n_samples=len(used),
        horizon_days=horizon,
        season_matched=season_matched,
    )


class SeriesEvidenceEstimator:
    """Builds the ``[series_estimator]`` evidence item for a forecast, if any.

    Applies only when the question metadata carries a benchmark series id the
    router understands AND a resolution date; everything else — unknown
    sources, missing dates, thin history, provider failures — degrades to no
    evidence, never a failed forecast.
    """

    def __init__(
        self,
        *,
        source: SeriesHistorySource,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        seasonal_window_days: int = DEFAULT_SEASONAL_WINDOW_DAYS,
    ) -> None:
        self._source = source
        self._min_samples = min_samples
        self._seasonal_window_days = seasonal_window_days

    def evidence(
        self, metadata: Mapping[str, Any] | None, *, as_of: datetime
    ) -> tuple[Evidence, ...]:
        if not metadata:
            return ()
        benchmark_id = metadata.get(BENCHMARK_QUESTION_ID_KEY)
        raw_resolution = metadata.get("resolution_date")
        if not benchmark_id or not raw_resolution:
            return ()
        try:
            resolution_date = ensure_utc(
                datetime.fromisoformat(str(raw_resolution).replace("Z", "+00:00"))
            )
        except ValueError:
            return ()

        ceiling = ensure_utc(as_of)
        history = self._source.history(str(benchmark_id), as_of=ceiling)
        estimate = estimate_direction_probability(
            history,
            as_of=ceiling,
            resolution_date=resolution_date,
            min_samples=self._min_samples,
            seasonal_window_days=self._seasonal_window_days,
        )
        if estimate is None:
            return ()
        for observed, _ in history:  # defense-in-depth over the provider contract
            if observed > ceiling.astimezone(UTC).date():
                msg = "series history contains an observation dated after the as-of ceiling."
                raise RuntimeError(msg)

        season_note = ", season-matched" if estimate.season_matched else ""
        snippet = (
            f"Deterministic estimate computed from this series' own history "
            f"(arithmetic, no model judgment): the probability that the value on "
            f"{resolution_date.date().isoformat()} is higher than at the forecast "
            f"time is approximately {estimate.probability:.2f}, based on "
            f"{estimate.n_samples} historical {estimate.horizon_days}-day "
            f"windows{season_note}. For direction questions about this series, "
            f"treat this as the reference-class base rate."
        )
        _LOG.info(
            "forecaster.series_estimate",
            benchmark_id=str(benchmark_id),
            probability=round(estimate.probability, 4),
            n_samples=estimate.n_samples,
            horizon_days=estimate.horizon_days,
            season_matched=estimate.season_matched,
        )
        return (
            Evidence(
                snippet=snippet,
                source=SERIES_ESTIMATOR_SOURCE,
                source_id=str(benchmark_id),
                knowledge_time=ceiling,
                score=1.0,
            ),
        )
