"""Network fetchers for benchmark question sets (C7).

Fetchers are the *only* part of the benchmark layer that touch the network. Each
one pulls raw records from a provider and emits the plain ``dict`` shape the
matching adapter's ``from_records`` expects, keeping the adapters pure and
hermetically testable. Fetch (network) and map (deterministic) stay separate so
the mapping is covered by unit tests without any live calls (CLAUDE.md §2.8).
"""

from __future__ import annotations

from benchmarks.fetchers.forecastbench_repo import ForecastBenchFetcher
from benchmarks.fetchers.metaculus_api import MetaculusFetcher

__all__ = ["ForecastBenchFetcher", "MetaculusFetcher"]
