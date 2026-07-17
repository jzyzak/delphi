"""As-of numeric series history providers (the quantitative evidence layer).

Fetches a series' historical observations so the deterministic
series-threshold estimator (``forecaster/stages/series_estimate.py``) can
compute crossing probabilities from the series' own past. The as-of discipline
(§2.1) holds the same way it does for GDELT/Wikipedia: observations carry
their own dates, and every provider hard-filters to ``date <= as_of`` before
returning — a post-as-of observation can never reach the estimator.

Providers are keyless public endpoints:
- FRED: ``fredgraph.csv`` (full history CSV, no API key)
- Yahoo chart API: daily closes for tickers
- DBnomics v22: official statistical series (weather, macro)
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

import structlog

from common.http.client import HttpClient

__all__ = [
    "DbnomicsSeriesProvider",
    "FredSeriesProvider",
    "SeriesHistoryProvider",
    "SeriesRouter",
    "YahooChartSeriesProvider",
    "parse_benchmark_series_ref",
]

_LOG = structlog.get_logger(__name__)


@runtime_checkable
class SeriesHistoryProvider(Protocol):
    """Historical observations of one numeric series, pinned at ``as_of``."""

    def history(self, ref: str, *, as_of: datetime) -> tuple[tuple[date, float], ...]:
        """Date-ascending ``(date, value)`` observations with ``date <= as_of``."""
        ...


def _as_of_date(as_of: datetime) -> date:
    return as_of.astimezone(UTC).date()


class FredSeriesProvider:
    """FRED series history via the keyless ``fredgraph.csv`` endpoint."""

    def __init__(
        self,
        *,
        http: HttpClient,
        base_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv",
    ) -> None:
        self._http = http
        self._base_url = base_url

    def history(self, ref: str, *, as_of: datetime) -> tuple[tuple[date, float], ...]:
        text = self._http.get_text(self._base_url, params={"id": ref})
        ceiling = _as_of_date(as_of)
        points: list[tuple[date, float]] = []
        for line in text.splitlines()[1:]:  # header: observation_date,<ID>
            parts = line.split(",")
            if len(parts) != 2:
                continue
            raw_date, raw_value = parts
            try:
                observed = date.fromisoformat(raw_date.strip())
                value = float(raw_value)
            except ValueError:
                continue  # FRED encodes missing values as "."
            if observed <= ceiling:
                points.append((observed, value))
        points.sort()
        return tuple(points)


class YahooChartSeriesProvider:
    """Daily closes for a ticker via Yahoo's chart endpoint (no key needed).

    Yahoo rate-limits non-browser user agents aggressively, so requests carry
    a browser-style UA (verified: the DELPHI UA gets 429, a Mozilla UA works).
    """

    _USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

    def __init__(
        self,
        *,
        http: HttpClient,
        base_url: str = "https://query1.finance.yahoo.com/v8/finance/chart",
        lookback: str = "10y",
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._lookback = lookback

    def history(self, ref: str, *, as_of: datetime) -> tuple[tuple[date, float], ...]:
        payload = self._http.get_json(
            f"{self._base_url}/{ref}",
            params={"range": self._lookback, "interval": "1d"},
            headers={"User-Agent": self._USER_AGENT},
        )
        result = _dig(payload, "chart", "result")
        if not isinstance(result, list) or not result:
            return ()
        first = result[0] if isinstance(result[0], Mapping) else {}
        timestamps = first.get("timestamp")
        quotes = _dig(first, "indicators", "quote")
        closes = (
            quotes[0].get("close")
            if isinstance(quotes, list) and quotes and isinstance(quotes[0], Mapping)
            else None
        )
        if not isinstance(timestamps, list) or not isinstance(closes, list):
            return ()
        ceiling = _as_of_date(as_of)
        points: list[tuple[date, float]] = []
        for ts, close in zip(timestamps, closes, strict=False):
            if close is None:
                continue
            try:
                observed = datetime.fromtimestamp(int(ts), tz=UTC).date()
                value = float(close)
            except (TypeError, ValueError, OSError):
                continue
            if observed <= ceiling:
                points.append((observed, value))
        points.sort()
        return tuple(points)


class DbnomicsSeriesProvider:
    """DBnomics v22 series history (``provider/dataset/series`` refs)."""

    def __init__(
        self,
        *,
        http: HttpClient,
        base_url: str = "https://api.db.nomics.world/v22/series",
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    def history(self, ref: str, *, as_of: datetime) -> tuple[tuple[date, float], ...]:
        payload = self._http.get_json(
            f"{self._base_url}/{ref}", params={"observations": "1", "format": "json"}
        )
        docs = _dig(payload, "series", "docs")
        if not isinstance(docs, list) or not docs or not isinstance(docs[0], Mapping):
            return ()
        periods = docs[0].get("period")
        values = docs[0].get("value")
        if not isinstance(periods, list) or not isinstance(values, list):
            return ()
        ceiling = _as_of_date(as_of)
        points: list[tuple[date, float]] = []
        for raw_period, raw_value in zip(periods, values, strict=False):
            if raw_value is None or raw_value == "NA":
                continue
            try:
                observed = date.fromisoformat(str(raw_period))
                value = float(raw_value)
            except ValueError:
                continue  # non-daily periods (e.g. "2024-Q1") are not supported
            if observed <= ceiling:
                points.append((observed, value))
        points.sort()
        return tuple(points)


def parse_benchmark_series_ref(benchmark_question_id: str) -> tuple[str, str] | None:
    """Split a benchmark question id into ``(source, provider_ref)``.

    Supported shapes (ForecastBench composite ids, ``benchmark:`` prefix
    optional): ``fred-DFF`` -> ``("fred", "DFF")``; ``yfinance-MMM`` ->
    ``("yfinance", "MMM")``; ``dbnomics-meteofrance_TEMPERATURE_celsius.07005.D``
    -> ``("dbnomics", "meteofrance/TEMPERATURE/celsius.07005.D")``. Returns
    ``None`` for anything else — unknown shapes must never be estimated.
    """
    composite = benchmark_question_id.rsplit(":", 1)[-1]
    source, sep, raw_ref = composite.partition("-")
    if not sep or not raw_ref:
        return None
    if source in ("fred", "yfinance"):
        return source, raw_ref
    if source == "dbnomics":
        parts = raw_ref.split("_", 2)
        if len(parts) != 3 or not all(parts):
            return None
        return source, "/".join(parts)
    return None


class SeriesRouter:
    """Routes a benchmark series source to its history provider."""

    def __init__(self, providers: Mapping[str, SeriesHistoryProvider]) -> None:
        self._providers = dict(providers)

    def history(
        self, benchmark_question_id: str, *, as_of: datetime
    ) -> tuple[tuple[date, float], ...]:
        """History for the series behind ``benchmark_question_id`` (or empty).

        Unknown sources and provider failures degrade to no data — the
        estimator then contributes nothing, and the forecast proceeds on the
        other evidence.
        """
        parsed = parse_benchmark_series_ref(benchmark_question_id)
        if parsed is None:
            return ()
        source, ref = parsed
        provider = self._providers.get(source)
        if provider is None:
            return ()
        try:
            return provider.history(ref, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - one series must not kill a forecast
            _LOG.warning(
                "sources.series.provider_failed",
                source=source,
                ref=ref,
                error=str(exc),
            )
            return ()


def _dig(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current
