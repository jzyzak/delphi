"""Concrete general-question forecast chain (composes the core building blocks).

App layer: wires the §3 pipeline (base rate -> decomposition -> inside view ->
ensemble -> aggregate -> supervisor -> calibrate -> uncertainty -> leakage judge)
into a single ``forecast(question, as_of)`` that writes a complete registry record.
"""

from __future__ import annotations

from forecaster.chain import Forecaster, ForecastResult

__all__ = ["ForecastResult", "Forecaster"]
