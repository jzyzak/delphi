"""Tests for the mandatory baselines (C6.3)."""

from __future__ import annotations

import pytest

from evaluation.baselines import (
    Baseline,
    clip_probability,
    market_consensus,
    strong_llm_baseline,
    superforecaster_median,
)


def test_superforecaster_median() -> None:
    assert superforecaster_median([0.2, 0.6, 0.4]) == pytest.approx(0.4)


def test_superforecaster_median_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        superforecaster_median([])


def test_market_and_llm_clip() -> None:
    assert market_consensus(0.0) > 0.0
    assert strong_llm_baseline(1.0) < 1.0
    assert market_consensus(0.5) == pytest.approx(0.5)


def test_clip_bounds() -> None:
    assert 0.0 < clip_probability(0.0) < clip_probability(1.0) < 1.0


def test_baseline_predict() -> None:
    b = Baseline(name="market", predictions={"q1": 0.7, "q2": 0.0})
    assert b.predict("q1") == pytest.approx(0.7)
    clipped = b.predict("q2")
    assert clipped is not None and clipped > 0.0
    assert b.predict("missing") is None
