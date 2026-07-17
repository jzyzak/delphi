"""Tests for the deterministic series-threshold estimator."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from forecaster.stages.series_estimate import (
    SERIES_ESTIMATOR_SOURCE,
    SeriesEvidenceEstimator,
    estimate_direction_probability,
)

AS_OF = datetime(2026, 3, 1, tzinfo=UTC)
RESOLUTION = datetime(2026, 3, 31, tzinfo=UTC)  # 30-day horizon


def _daily(
    values: list[float], *, start: date = date(2025, 1, 1)
) -> tuple[tuple[date, float], ...]:
    return tuple((start + timedelta(days=i), v) for i, v in enumerate(values))


class TestEstimateDirectionProbability:
    def test_rising_series_yields_high_probability(self) -> None:
        history = _daily([float(i) for i in range(400)])
        estimate = estimate_direction_probability(history, as_of=AS_OF, resolution_date=RESOLUTION)
        assert estimate is not None
        assert estimate.horizon_days == 30
        assert estimate.probability > 0.95
        # Laplace smoothing keeps the estimate off the poles.
        assert estimate.probability < 1.0

    def test_falling_series_yields_low_probability(self) -> None:
        history = _daily([float(-i) for i in range(400)])
        estimate = estimate_direction_probability(history, as_of=AS_OF, resolution_date=RESOLUTION)
        assert estimate is not None
        assert 0.0 < estimate.probability < 0.05

    def test_flat_series_counts_no_rises(self) -> None:
        history = _daily([5.0] * 400)
        estimate = estimate_direction_probability(history, as_of=AS_OF, resolution_date=RESOLUTION)
        assert estimate is not None
        # Strict "higher than": a flat series never rises.
        assert estimate.probability == pytest.approx(1 / (estimate.n_samples + 2))

    def test_alternating_series_is_near_half(self) -> None:
        history = _daily([float(i % 2) for i in range(400)])
        # An odd horizon flips the alternating series every window: exactly
        # half the windows rise (an even horizon would produce all ties).
        estimate = estimate_direction_probability(
            history, as_of=AS_OF, resolution_date=AS_OF + timedelta(days=31)
        )
        assert estimate is not None
        assert 0.4 < estimate.probability < 0.6

    def test_non_positive_horizon_returns_none(self) -> None:
        history = _daily([1.0] * 100)
        assert estimate_direction_probability(history, as_of=AS_OF, resolution_date=AS_OF) is None
        assert (
            estimate_direction_probability(
                history, as_of=AS_OF, resolution_date=AS_OF - timedelta(days=5)
            )
            is None
        )

    def test_thin_history_returns_none(self) -> None:
        history = _daily([1.0] * 40)  # only ~10 thirty-day windows
        assert (
            estimate_direction_probability(history, as_of=AS_OF, resolution_date=RESOLUTION) is None
        )
        assert estimate_direction_probability((), as_of=AS_OF, resolution_date=RESOLUTION) is None

    def test_sparse_series_beyond_tolerance_returns_none(self) -> None:
        # Monthly-ish gaps larger than the match tolerance: no valid windows.
        history = tuple((date(2020, 1, 1) + timedelta(days=45 * i), float(i)) for i in range(40))
        assert (
            estimate_direction_probability(
                history, as_of=AS_OF, resolution_date=AS_OF + timedelta(days=10)
            )
            is None
        )

    def test_seasonal_series_is_season_matched(self) -> None:
        # Four years of daily data: values rise during Feb-Apr windows and fall
        # the rest of the year. Season-matching around a March 1 as-of must
        # pick up the rising regime, where the all-year rate would be ~50/50.
        points: list[tuple[date, float]] = []
        value = 100.0
        start = date(2022, 1, 1)
        for i in range(365 * 4):
            observed = start + timedelta(days=i)
            rising_season = 32 <= observed.timetuple().tm_yday <= 130
            value += 1.0 if rising_season else -0.35
            points.append((observed, value))
        estimate = estimate_direction_probability(
            tuple(points), as_of=AS_OF, resolution_date=RESOLUTION
        )
        assert estimate is not None
        assert estimate.season_matched is True
        assert estimate.probability > 0.8

    def test_falls_back_to_all_samples_when_season_is_thin(self) -> None:
        # Barely more than one year: few windows share the as-of's season, so
        # the estimator uses the full sample instead.
        history = _daily([float(i) for i in range(400)])
        estimate = estimate_direction_probability(
            history,
            as_of=AS_OF,
            resolution_date=RESOLUTION,
            seasonal_window_days=2,
        )
        assert estimate is not None
        assert estimate.season_matched is False


class _FixedSource:
    def __init__(self, history: tuple[tuple[date, float], ...]) -> None:
        self._history = history
        self.calls: list[tuple[str, datetime]] = []

    def history(self, benchmark_question_id: str, *, as_of: datetime):
        self.calls.append((benchmark_question_id, as_of))
        return self._history


def _metadata(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "benchmark_question_id": "forecastbench:fred-DFF",
        "resolution_date": RESOLUTION.isoformat(),
    }
    base.update(overrides)
    return base


class TestSeriesEvidenceEstimator:
    def test_builds_evidence_from_history(self) -> None:
        source = _FixedSource(_daily([float(i) for i in range(400)]))
        estimator = SeriesEvidenceEstimator(source=source)
        items = estimator.evidence(_metadata(), as_of=AS_OF)
        assert len(items) == 1
        item = items[0]
        assert item.source == SERIES_ESTIMATOR_SOURCE
        assert item.source_id == "forecastbench:fred-DFF"
        assert item.knowledge_time == AS_OF  # as-of-safe by construction
        assert "30-day windows" in item.snippet
        assert "0.9" in item.snippet  # rising series -> high probability
        assert source.calls == [("forecastbench:fred-DFF", AS_OF)]

    def test_no_metadata_or_missing_fields_yield_nothing(self) -> None:
        estimator = SeriesEvidenceEstimator(source=_FixedSource(()))
        assert estimator.evidence(None, as_of=AS_OF) == ()
        assert estimator.evidence({}, as_of=AS_OF) == ()
        assert estimator.evidence(_metadata(benchmark_question_id=""), as_of=AS_OF) == ()
        assert estimator.evidence(_metadata(resolution_date=""), as_of=AS_OF) == ()

    def test_unparseable_resolution_date_yields_nothing(self) -> None:
        estimator = SeriesEvidenceEstimator(source=_FixedSource(()))
        assert estimator.evidence(_metadata(resolution_date="soon"), as_of=AS_OF) == ()

    def test_thin_history_yields_nothing(self) -> None:
        estimator = SeriesEvidenceEstimator(source=_FixedSource(_daily([1.0] * 5)))
        assert estimator.evidence(_metadata(), as_of=AS_OF) == ()

    def test_post_as_of_observation_raises(self) -> None:
        # Defense-in-depth over the provider contract (§2.1): a misbehaving
        # source that leaks a future observation must fail loudly, not forecast.
        leaky = _daily([float(i) for i in range(400)]) + ((date(2026, 4, 1), 999.0),)
        estimator = SeriesEvidenceEstimator(source=_FixedSource(leaky))
        with pytest.raises(RuntimeError, match="after the as-of ceiling"):
            estimator.evidence(_metadata(), as_of=AS_OF)

    def test_zulu_resolution_date_is_accepted(self) -> None:
        source = _FixedSource(_daily([float(i) for i in range(400)]))
        estimator = SeriesEvidenceEstimator(source=source)
        metadata = _metadata(resolution_date="2026-03-31T00:00:00Z")
        assert len(estimator.evidence(metadata, as_of=AS_OF)) == 1
