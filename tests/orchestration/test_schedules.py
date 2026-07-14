"""Unit tests for orchestration schedules (§8)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.orchestration.schedules import Cadence, default_orchestration_config, default_schedules


def test_cadence_schedule_expression_hours() -> None:
    cadence = Cadence(interval=timedelta(hours=6))
    assert cadence.schedule_expression == "rate(6 hours)"


def test_cadence_schedule_expression_minutes() -> None:
    cadence = Cadence(interval=timedelta(minutes=15))
    assert cadence.schedule_expression == "rate(15 minutes)"


def test_cadence_schedule_expression_days() -> None:
    cadence = Cadence(interval=timedelta(days=1))
    assert cadence.schedule_expression == "rate(1 day)"


def test_cadence_invalid_interval_raises() -> None:
    cadence = Cadence(interval=timedelta(seconds=0))
    with pytest.raises(ValueError, match="interval must be positive"):
        _ = cadence.schedule_expression


def test_default_schedules_has_all_loops() -> None:
    schedules = default_schedules()
    assert schedules.research.interval == timedelta(hours=6)
    assert schedules.allocation.interval == timedelta(hours=1)
    assert schedules.monitoring.interval == timedelta(minutes=15)


def test_default_orchestration_config() -> None:
    config = default_orchestration_config()
    assert config.global_trials_budget == 100
    assert config.submissions_per_agent == 3
