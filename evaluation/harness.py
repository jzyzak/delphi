"""Guarded-set eval harness: trials ledger + holdout governor (C6.6).

THE HOUSE (CLAUDE.md §2.2/§2.4). Every evaluation against a guarded set debits
the append-only trials ledger; silently trying many variants and reporting the
best is method-overfitting, the central anti-pattern, and the ledger exists to
make it visible. Holdout data is reachable ONLY through the logged, budgeted
governor. Both gates fail *closed*: if the ledger has no budget, or the holdout
governor is absent, the evaluation raises rather than proceeding unrecorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.orchestration.budget import BudgetLedger, BudgetSnapshot
from core.orchestration.meta.holdout import HoldoutGovernor
from evaluation.aggregate import ScoreSummary, summarize_scores
from evaluation.scoring import ScoredRecord, Scorer

__all__ = ["EvalHarness", "HoldoutUnavailable", "TrialsLedgerExhausted"]


class TrialsLedgerExhausted(RuntimeError):
    """Raised when a guarded-set evaluation cannot draw down the trials budget."""


class HoldoutUnavailable(RuntimeError):
    """Raised when holdout access is attempted without a configured governor."""


class EvalHarness:
    """Runs guarded-set evaluations, debiting the ledger on every run."""

    def __init__(
        self,
        *,
        budget_ledger: BudgetLedger,
        holdout: HoldoutGovernor | None = None,
    ) -> None:
        self._ledger = budget_ledger
        self._holdout = holdout

    def evaluate_guarded(
        self,
        scorer: Scorer,
        records: Sequence[ScoredRecord],
        *,
        n_boot: int = 1000,
        seed: int = 0,
    ) -> ScoreSummary:
        """Score a guarded set, debiting one trial per question (fail-closed)."""
        if not records:
            msg = "cannot evaluate an empty guarded set."
            raise ValueError(msg)
        grant = self._ledger.reserve_budget(n=len(records))
        if grant is None:
            msg = (
                "Trials budget exhausted: refusing to evaluate the guarded set. "
                "Every guarded evaluation must be recorded (CLAUDE.md §2.4)."
            )
            raise TrialsLedgerExhausted(msg)
        try:
            summary = summarize_scores(scorer, records, n_boot=n_boot, seed=seed)
        except Exception:
            self._ledger.release(grant)
            raise
        self._ledger.commit(grant)
        return summary

    def evaluate_guarded_scorers(
        self,
        scorers: Sequence[Scorer],
        records: Sequence[ScoredRecord],
        *,
        n_boot: int = 1000,
        seed: int = 0,
    ) -> dict[str, ScoreSummary]:
        """Score a guarded set with several scorers in ONE ledger draw (fail-closed)."""
        if not records:
            msg = "cannot evaluate an empty guarded set."
            raise ValueError(msg)
        if not scorers:
            msg = "at least one scorer is required."
            raise ValueError(msg)
        grant = self._ledger.reserve_budget(n=len(records))
        if grant is None:
            msg = "Trials budget exhausted: refusing to evaluate the guarded set."
            raise TrialsLedgerExhausted(msg)
        try:
            summaries = {
                scorer.name: summarize_scores(scorer, records, n_boot=n_boot, seed=seed)
                for scorer in scorers
            }
        except Exception:
            self._ledger.release(grant)
            raise
        self._ledger.commit(grant)
        return summaries

    def ledger_snapshot(self) -> BudgetSnapshot:
        """Observed trials-ledger usage, for the rendered report (§2.4).

        Read-only observability — never a path around ``reserve_budget``.
        """
        return self._ledger.snapshot()

    @property
    def ledger_durable(self) -> bool:
        """Whether the ledger's committed count survives this process (§2.4)."""
        return self._ledger.durable

    def access_holdout(self, *, reason: str) -> Mapping[str, Any]:
        """Access the holdout ONLY through the governor (logged + budgeted)."""
        if self._holdout is None:
            msg = "No holdout governor configured; there is no unlogged path (§2.2)."
            raise HoldoutUnavailable(msg)
        return self._holdout.access_holdout(reason=reason).payload
