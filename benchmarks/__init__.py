"""Benchmark adapters: ForecastBench / Metaculus / market-consensus / live.

App layer: map external question sets into DELPHI's intake/resolution shapes with
strict as-of pinning, and feed them through the eval harness.
"""

from __future__ import annotations

from benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkQuestion,
    BenchmarkResolution,
    as_of_pins,
    assert_no_leakage,
    scored_records,
)
from benchmarks.forecastbench import ForecastBenchAdapter
from benchmarks.live import LiveHarvestAdapter
from benchmarks.market_consensus import MarketConsensusAdapter, consensus_baseline
from benchmarks.metaculus import MetaculusAdapter

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkQuestion",
    "BenchmarkResolution",
    "ForecastBenchAdapter",
    "LiveHarvestAdapter",
    "MarketConsensusAdapter",
    "MetaculusAdapter",
    "as_of_pins",
    "assert_no_leakage",
    "consensus_baseline",
    "scored_records",
]
