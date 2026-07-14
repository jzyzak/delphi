"""Forecast-chain stages (CLAUDE.md §3).

Each stage is a thin, testable function/dataclass over the shared ``core/forecast``
building blocks: reference class / base rate -> decomposition -> inside view +
ensemble -> aggregate + supervisor -> calibrate + uncertainty -> leakage gate.
Every stage takes ``as_of`` as an explicit input; none calls ``now()``.
"""

from __future__ import annotations
