"""Shared orchestration types."""

from __future__ import annotations

from enum import StrEnum


class LoopName(StrEnum):
    """Named orchestration loops."""

    RESEARCH = "research"
    ALLOCATION = "allocation"
    MONITORING = "monitoring"


class StepStatus(StrEnum):
    """Persisted status of one orchestration step."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ClaimResult(StrEnum):
    """Outcome of attempting to claim a step for execution."""

    CLAIMED = "claimed"
    ALREADY_SUCCEEDED = "already_succeeded"
