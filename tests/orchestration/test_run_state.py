"""Tests for restart-safe run-state store (O2, O4 + §8)."""

from __future__ import annotations

import pytest

from core.orchestration.run_state import InMemoryRunStateStore
from core.orchestration.types import ClaimResult, LoopName, StepStatus
from tests.orchestration.conftest import CLOCK_START

pytestmark_postgres = pytest.mark.postgres


def test_claim_new_step(memory_run_state: InMemoryRunStateStore) -> None:
    result = memory_run_state.claim_step(
        step_id="research:2025-01-01T12:00:00+00:00",
        loop_name=LoopName.RESEARCH,
        tick_at=CLOCK_START,
    )
    assert result is ClaimResult.CLAIMED


def test_o2_already_succeeded_skips(memory_run_state: InMemoryRunStateStore) -> None:
    step_id = "research:2025-01-01T12:00:00+00:00"
    memory_run_state.claim_step(
        step_id=step_id,
        loop_name=LoopName.RESEARCH,
        tick_at=CLOCK_START,
    )
    memory_run_state.mark_succeeded(step_id=step_id)
    result = memory_run_state.claim_step(
        step_id=step_id,
        loop_name=LoopName.RESEARCH,
        tick_at=CLOCK_START,
    )
    assert result is ClaimResult.ALREADY_SUCCEEDED


def test_o4_failed_step_can_be_reclaimed(memory_run_state: InMemoryRunStateStore) -> None:
    step_id = "allocation:2025-01-01T12:00:00+00:00"
    memory_run_state.claim_step(
        step_id=step_id,
        loop_name=LoopName.ALLOCATION,
        tick_at=CLOCK_START,
    )
    memory_run_state.mark_failed(step_id=step_id, error_message="boom")
    result = memory_run_state.claim_step(
        step_id=step_id,
        loop_name=LoopName.ALLOCATION,
        tick_at=CLOCK_START,
    )
    assert result is ClaimResult.CLAIMED
    record = memory_run_state.get_step(step_id)
    assert record is not None
    assert record.status is StepStatus.RUNNING


@pytest.mark.postgres
def test_postgres_claim_and_succeed(postgres_run_state) -> None:
    step_id = "monitoring:2025-01-01T12:00:00+00:00"
    assert (
        postgres_run_state.claim_step(
            step_id=step_id,
            loop_name=LoopName.MONITORING,
            tick_at=CLOCK_START,
        )
        is ClaimResult.CLAIMED
    )
    postgres_run_state.mark_succeeded(step_id=step_id)
    assert (
        postgres_run_state.claim_step(
            step_id=step_id,
            loop_name=LoopName.MONITORING,
            tick_at=CLOCK_START,
        )
        is ClaimResult.ALREADY_SUCCEEDED
    )
