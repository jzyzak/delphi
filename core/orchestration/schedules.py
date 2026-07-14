"""Loop cadences and orchestration configuration."""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field

from core.orchestration.types import LoopName


class Cadence(BaseModel):
    """Scheduling interval for one orchestration loop."""

    model_config = ConfigDict(frozen=True)

    interval: timedelta = Field(description="Minimum time between loop invocations.")

    @property
    def schedule_expression(self) -> str:
        """EventBridge-compatible rate expression (config-only; no AWS wiring)."""
        total_seconds = int(self.interval.total_seconds())
        if total_seconds <= 0:
            msg = "interval must be positive."
            raise ValueError(msg)
        if total_seconds % 86_400 == 0:
            days = total_seconds // 86_400
            return f"rate({days} day{'s' if days != 1 else ''})"
        if total_seconds % 3_600 == 0:
            hours = total_seconds // 3_600
            return f"rate({hours} hour{'s' if hours != 1 else ''})"
        if total_seconds % 60 == 0:
            minutes = total_seconds // 60
            return f"rate({minutes} minute{'s' if minutes != 1 else ''})"
        return f"rate({total_seconds} seconds)"


class LoopSchedules(BaseModel):
    """Per-loop cadence configuration."""

    model_config = ConfigDict(frozen=True)

    research: Cadence
    allocation: Cadence
    monitoring: Cadence

    def cadence_for(self, loop: LoopName) -> Cadence:
        return getattr(self, loop.value)


class OrchestrationConfig(BaseModel):
    """Top-level orchestration parameters."""

    model_config = ConfigDict(frozen=True)

    global_trials_budget: int = Field(ge=0)
    submissions_per_agent: int = Field(default=1, ge=1)
    schedules: LoopSchedules
    tick_seconds: float = Field(default=1.0, gt=0.0, description="Scheduler poll interval.")


def default_schedules() -> LoopSchedules:
    """Default loop cadences aligned with CLAUDE.md thin-orchestrator guidance."""
    return LoopSchedules(
        research=Cadence(interval=timedelta(hours=6)),
        allocation=Cadence(interval=timedelta(hours=1)),
        monitoring=Cadence(interval=timedelta(minutes=15)),
    )


def default_orchestration_config() -> OrchestrationConfig:
    """Default orchestration configuration for local and paper-trading runs."""
    return OrchestrationConfig(
        global_trials_budget=100,
        submissions_per_agent=3,
        schedules=default_schedules(),
    )
