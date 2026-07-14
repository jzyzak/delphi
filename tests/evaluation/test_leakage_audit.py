"""Tests for the suite-level leakage audit (C6.7)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.forecast.leakage_judge import (
    FixtureLeakageJudgeLLM,
    LeakageJudge,
    Trace,
    TraceComponent,
)
from evaluation.leakage_audit import audit_suite

_AS_OF = datetime(2026, 1, 1, tzinfo=UTC)


def _trace(text: str) -> Trace:
    return Trace(component=TraceComponent.SEARCH, as_of=_AS_OF, text=text, forecast_id="f1")


def _judge() -> LeakageJudge:
    return LeakageJudge(
        FixtureLeakageJudgeLLM(flag_substrings=("LEAK",), reject_future_iso_dates=False)
    )


def test_leakage_rate_and_clean_fraction() -> None:
    traces = [_trace("clean evidence"), _trace("this trace LEAKs the future")]
    audit = audit_suite(_judge(), traces)
    assert audit.leakage_rate == pytest.approx(0.5)
    assert audit.clean_fraction == pytest.approx(0.5)
    assert audit.report.total == 2
    assert audit.report.flagged == 1


def test_clean_suite_zero_rate() -> None:
    audit = audit_suite(_judge(), [_trace("nothing to see"), _trace("also clean")])
    assert audit.leakage_rate == pytest.approx(0.0)
    assert audit.clean_fraction == pytest.approx(1.0)


def test_render_contains_rate() -> None:
    rendered = audit_suite(_judge(), [_trace("LEAK")]).render()
    assert "leakage_rate" in rendered
    assert "clean_fraction" in rendered
