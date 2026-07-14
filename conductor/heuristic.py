"""Heuristic conductor (C8.2) — ships as the product (CLAUDE.md §4, Stage 1).

A hand-designed, deterministic orchestration over the role set that runs the
fixed §3 chain, injects a red-team counter-case, and applies a verifier
accept/revise loop. It is designed to *match or beat* the fixed chain, never
underperform it: the verifier only requests a revision when the leakage gate
quarantined the forecast, and otherwise accepts the chain's forecast unchanged.
Every run records a complete, auditable workflow trace (provenance is preserved,
never hidden — §4/§9), which becomes the Stage-2 training corpus (C8.3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from conductor.roles import ROLE_CONTRACTS, RoleId
from core.pit.models import ensure_utc
from forecaster.chain import Forecaster, ForecastResult

__all__ = [
    "ConductorResult",
    "FixtureRedTeamLLM",
    "HeuristicConductor",
    "RedTeamLLM",
    "WorkflowStep",
    "WorkflowTrace",
]

# Deterministic route policy over the role set (C8.2.a).
_ROUTE: tuple[RoleId, ...] = (
    RoleId.RESEARCHER,
    RoleId.REFERENCE_CLASS,
    RoleId.DECOMPOSER,
    RoleId.ESTIMATOR,
    RoleId.AGGREGATOR,
    RoleId.RED_TEAM,
    RoleId.VERIFIER,
    RoleId.CALIBRATOR,
)


@runtime_checkable
class RedTeamLLM(Protocol):
    """Mockable seam for the red-team's counter-case."""

    @property
    def model_version(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def challenge(self, *, question: str, probability: float, as_of: datetime) -> str:
        """Return the strongest counter-argument against ``probability``."""
        ...


class FixtureRedTeamLLM:
    """Deterministic red-team for tests (no network)."""

    model_version = "fixture-red-team-v1"
    prompt_version = "v1"

    def __init__(self, *, counter: str = "") -> None:
        self._counter = counter

    def challenge(self, *, question: str, probability: float, as_of: datetime) -> str:
        if self._counter:
            return self._counter
        return (
            f"Counter-case: the {probability:.2f} estimate may over-weight the "
            f"available evidence for '{question}'."
        )


@dataclass(frozen=True)
class WorkflowStep:
    """One executed role: its id, access list, and a short summary."""

    role_id: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class WorkflowTrace:
    """The full ordered workflow the conductor executed (provenance, §4)."""

    steps: tuple[WorkflowStep, ...]
    route: tuple[str, ...]
    revisions: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "route": list(self.route),
            "revisions": self.revisions,
            "steps": [
                {
                    "role_id": s.role_id,
                    "reads": list(s.reads),
                    "writes": list(s.writes),
                    "summary": s.summary,
                }
                for s in self.steps
            ],
        }


@dataclass(frozen=True)
class ConductorResult:
    """The conductor's output: the forecast + workflow trace + red-team counter."""

    forecast: ForecastResult
    workflow: WorkflowTrace
    red_team_counter: str = ""
    verifier_accepted: bool = True
    revisions: int = 0


def _step(role_id: RoleId, summary: str) -> WorkflowStep:
    contract = ROLE_CONTRACTS[role_id]
    return WorkflowStep(
        role_id=role_id.value,
        reads=tuple(sorted(contract.access.reads)),
        writes=tuple(sorted(contract.access.writes)),
        summary=summary,
    )


class HeuristicConductor:
    """Deterministic role orchestration over the fixed forecast chain."""

    def __init__(
        self,
        *,
        forecaster: Forecaster,
        red_team: RedTeamLLM | None = None,
        max_revisions: int = 1,
    ) -> None:
        if max_revisions < 0:
            msg = "max_revisions must be >= 0."
            raise ValueError(msg)
        self._forecaster = forecaster
        self._red_team = red_team if red_team is not None else FixtureRedTeamLLM()
        self._max_revisions = max_revisions

    @property
    def route(self) -> tuple[RoleId, ...]:
        return _ROUTE

    def conduct(
        self,
        question: str,
        *,
        as_of: datetime,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConductorResult:
        """Run the routed workflow and return the forecast + its trace.

        ``metadata`` is passed through to the forecast chain and recorded on the
        question (e.g. a benchmark question id used by live-loop resolution).
        """
        ceiling = ensure_utc(as_of)
        result = self._forecaster.forecast(question, as_of=ceiling, metadata=metadata)

        if not result.accepted or result.probability is None:
            steps = (_step(RoleId.RESEARCHER, "intake refused the question; no forecast formed"),)
            return ConductorResult(
                forecast=result,
                workflow=WorkflowTrace(steps=steps, route=(RoleId.RESEARCHER.value,)),
                verifier_accepted=False,
            )

        # Verifier accept/revise loop (C8.2.c): revise only when the leakage gate
        # quarantined the forecast, so the conductor can never do worse than the
        # fixed chain on a clean run.
        revisions = 0
        while result.quarantined and revisions < self._max_revisions:
            revisions += 1
            result = self._forecaster.forecast(question, as_of=ceiling, metadata=metadata)
        verifier_accepted = not result.quarantined
        assert result.probability is not None  # re-forecast of an accepted question stays accepted

        counter = self._red_team.challenge(
            question=question, probability=result.probability, as_of=ceiling
        )

        steps = self._build_steps(result, counter=counter, verifier_accepted=verifier_accepted)
        workflow = WorkflowTrace(
            steps=steps, route=tuple(r.value for r in _ROUTE), revisions=revisions
        )
        return ConductorResult(
            forecast=result,
            workflow=workflow,
            red_team_counter=counter,
            verifier_accepted=verifier_accepted,
            revisions=revisions,
        )

    def _build_steps(
        self, result: ForecastResult, *, counter: str, verifier_accepted: bool
    ) -> tuple[WorkflowStep, ...]:
        summaries: dict[RoleId, str] = {
            RoleId.RESEARCHER: f"retrieved {len(result.evidence)} as-of evidence item(s)",
            RoleId.REFERENCE_CLASS: "established reference class + base rate",
            RoleId.DECOMPOSER: "decomposed into sub-questions",
            RoleId.ESTIMATOR: "assembled the method-agent ensemble",
            RoleId.AGGREGATOR: "aggregated + reconciled disagreement",
            RoleId.RED_TEAM: f"counter-case: {counter}",
            RoleId.VERIFIER: (
                "accepted" if verifier_accepted else "flagged by leakage gate (quarantined)"
            ),
            RoleId.CALIBRATOR: f"calibrated probability {result.probability:.3f}",
        }
        return tuple(_step(role_id, summaries[role_id]) for role_id in _ROUTE)

    def model_provenance(self) -> dict[str, object]:
        """Conductor-level provenance (red-team model + route policy)."""
        return {
            "conductor": "heuristic",
            "route": [r.value for r in _ROUTE],
            "red_team": {
                "model_version": self._red_team.model_version,
                "prompt_version": self._red_team.prompt_version,
            },
        }
