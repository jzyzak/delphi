"""Tests for the as-of series history providers (hermetic; network mocked)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx

from common.http.client import HttpClient
from sources.series import (
    DbnomicsSeriesProvider,
    FredSeriesProvider,
    SeriesHistoryProvider,
    SeriesRouter,
    YahooChartSeriesProvider,
    parse_benchmark_series_ref,
)

AS_OF = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _http(handler: Any) -> HttpClient:
    return HttpClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestFredSeriesProvider:
    def test_parses_filters_and_sorts(self) -> None:
        csv = (
            "observation_date,DFF\n"
            "2026-02-27,4.33\n"
            "2026-02-26,4.30\n"
            "2026-03-02,9.99\n"  # post-as-of: must never be returned (§2.1)
            "2026-02-28,.\n"  # FRED missing-value sentinel
            "garbage-line\n"
        )
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["params"] = dict(req.url.params)
            return httpx.Response(200, text=csv)

        provider = FredSeriesProvider(http=_http(handler))
        history = provider.history("DFF", as_of=AS_OF)
        assert seen["params"] == {"id": "DFF"}
        assert history == (
            (date(2026, 2, 26), 4.30),
            (date(2026, 2, 27), 4.33),
        )
        assert all(observed <= AS_OF.date() for observed, _ in history)


class TestYahooChartSeriesProvider:
    def _payload(self, timestamps: list[int], closes: list[float | None]) -> dict[str, Any]:
        return {
            "chart": {
                "result": [{"timestamp": timestamps, "indicators": {"quote": [{"close": closes}]}}]
            }
        }

    def test_parses_filters_and_skips_nulls(self) -> None:
        pre = int(datetime(2026, 2, 27, 21, tzinfo=UTC).timestamp())
        also_pre = int(datetime(2026, 2, 26, 21, tzinfo=UTC).timestamp())
        post = int(datetime(2026, 3, 3, 21, tzinfo=UTC).timestamp())
        seen: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["ua"] = req.headers.get("user-agent")
            return httpx.Response(
                200, json=self._payload([also_pre, pre, post], [100.0, None, 999.0])
            )

        provider = YahooChartSeriesProvider(http=_http(handler))
        history = provider.history("MMM", as_of=AS_OF)
        # The null close is skipped; the post-as-of bar is filtered (§2.1).
        assert history == ((date(2026, 2, 26), 100.0),)
        # Yahoo 429s non-browser user agents; the provider sends a browser UA.
        assert seen["ua"] is not None and seen["ua"].startswith("Mozilla/5.0")

    def test_malformed_payload_yields_empty(self) -> None:
        for payload in ({}, {"chart": {"result": []}}, {"chart": {"result": [{"timestamp": 3}]}}):
            provider = YahooChartSeriesProvider(
                http=_http(lambda _r, p=payload: httpx.Response(200, json=p))
            )
            assert provider.history("MMM", as_of=AS_OF) == ()


class TestDbnomicsSeriesProvider:
    def test_parses_filters_and_skips_na(self) -> None:
        payload = {
            "series": {
                "docs": [
                    {
                        "period": [
                            "2026-02-26",
                            "2026-02-27",
                            "2026-02-28",
                            "2026-03-05",
                            "2024-Q1",
                        ],
                        "value": [3.5, "NA", None, 9.9, 1.0],
                    }
                ]
            }
        }
        provider = DbnomicsSeriesProvider(http=_http(lambda _r: httpx.Response(200, json=payload)))
        history = provider.history("meteofrance/TEMPERATURE/celsius.07005.D", as_of=AS_OF)
        # NA and null skipped, post-as-of filtered (§2.1), quarterly period skipped.
        assert history == ((date(2026, 2, 26), 3.5),)

    def test_malformed_payload_yields_empty(self) -> None:
        for payload in ({}, {"series": {"docs": []}}, {"series": {"docs": [{"period": 1}]}}):
            provider = DbnomicsSeriesProvider(
                http=_http(lambda _r, p=payload: httpx.Response(200, json=p))
            )
            assert provider.history("a/b/c", as_of=AS_OF) == ()


class TestParseBenchmarkSeriesRef:
    def test_known_shapes(self) -> None:
        assert parse_benchmark_series_ref("forecastbench:fred-DFF") == ("fred", "DFF")
        assert parse_benchmark_series_ref("fred-T10Y3M") == ("fred", "T10Y3M")
        assert parse_benchmark_series_ref("forecastbench:yfinance-MMM") == ("yfinance", "MMM")
        assert parse_benchmark_series_ref(
            "forecastbench:dbnomics-meteofrance_TEMPERATURE_celsius.07005.D"
        ) == ("dbnomics", "meteofrance/TEMPERATURE/celsius.07005.D")

    def test_unknown_and_malformed_shapes(self) -> None:
        assert parse_benchmark_series_ref("forecastbench:acled-abc123") is None
        assert parse_benchmark_series_ref("forecastbench:polymarket-0xdeadbeef") is None
        assert parse_benchmark_series_ref("nodash") is None
        assert parse_benchmark_series_ref("fred-") is None
        assert parse_benchmark_series_ref("dbnomics-only_two") is None
        assert parse_benchmark_series_ref("dbnomics-a__b") is None  # empty middle segment


class TestSeriesRouter:
    def test_routes_to_the_matching_provider(self) -> None:
        class _Fixed:
            def __init__(self) -> None:
                self.calls: list[tuple[str, datetime]] = []

            def history(self, ref: str, *, as_of: datetime) -> tuple[tuple[date, float], ...]:
                self.calls.append((ref, as_of))
                return ((date(2026, 2, 1), 1.0),)

        fixed = _Fixed()
        assert isinstance(fixed, SeriesHistoryProvider)
        router = SeriesRouter({"fred": fixed})
        history = router.history("forecastbench:fred-DFF", as_of=AS_OF)
        assert history == ((date(2026, 2, 1), 1.0),)
        assert fixed.calls == [("DFF", AS_OF)]

    def test_unknown_source_yields_empty(self) -> None:
        router = SeriesRouter({})
        assert router.history("forecastbench:fred-DFF", as_of=AS_OF) == ()
        assert router.history("forecastbench:acled-abc", as_of=AS_OF) == ()

    def test_provider_failure_degrades_to_empty(self) -> None:
        class _Boom:
            def history(self, ref: str, *, as_of: datetime) -> tuple[tuple[date, float], ...]:
                msg = "provider exploded"
                raise RuntimeError(msg)

        router = SeriesRouter({"fred": _Boom()})
        assert router.history("forecastbench:fred-DFF", as_of=AS_OF) == ()
