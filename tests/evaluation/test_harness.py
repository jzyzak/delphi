"""Tests for the guarded-set harness: ledger debit + holdout governor (C6.6)."""

from __future__ import annotations

import pytest

from core.orchestration.budget import InMemoryBudgetLedger
from core.orchestration.meta.holdout import InMemoryHoldoutGovernor, StaticHoldoutSource
from evaluation.harness import EvalHarness, HoldoutUnavailable, TrialsLedgerExhausted
from evaluation.scoring import BrierScorer, LogScorer, ScoredRecord


class _FailingScorer:
    name = "boom"

    def score(self, prediction: float, outcome: float) -> float:
        msg = "scorer exploded"
        raise RuntimeError(msg)


def _records(n: int) -> list[ScoredRecord]:
    return [
        ScoredRecord(question_id=f"q{i}", domain="d", probability=0.5, outcome=float(i % 2))
        for i in range(n)
    ]


def _ledger(cap: int) -> InMemoryBudgetLedger:
    return InMemoryBudgetLedger(cap=cap, trials_count=lambda: 0)


class TestEvaluateGuarded:
    def test_scores_and_commits(self) -> None:
        ledger = _ledger(10)
        harness = EvalHarness(budget_ledger=ledger)
        summary = harness.evaluate_guarded(BrierScorer(), _records(4), n_boot=50)
        assert summary.n == 4
        # Reservation committed -> nothing left outstanding.
        assert ledger.snapshot().outstanding_reserved == 0

    def test_fail_closed_when_budget_exhausted(self) -> None:
        harness = EvalHarness(budget_ledger=_ledger(1))
        with pytest.raises(TrialsLedgerExhausted):
            harness.evaluate_guarded(BrierScorer(), _records(3))

    def test_release_on_scorer_failure(self) -> None:
        ledger = _ledger(10)
        harness = EvalHarness(budget_ledger=ledger)
        with pytest.raises(RuntimeError, match="exploded"):
            harness.evaluate_guarded(_FailingScorer(), _records(3))
        assert ledger.snapshot().outstanding_reserved == 0

    def test_empty_records_raises(self) -> None:
        harness = EvalHarness(budget_ledger=_ledger(10))
        with pytest.raises(ValueError, match="empty guarded set"):
            harness.evaluate_guarded(BrierScorer(), [])


class TestEvaluateGuardedScorers:
    def test_multi_scorer_single_draw(self) -> None:
        ledger = _ledger(10)
        harness = EvalHarness(budget_ledger=ledger)
        summaries = harness.evaluate_guarded_scorers(
            [BrierScorer(), LogScorer()], _records(4), n_boot=50
        )
        assert set(summaries) == {"brier", "log"}
        assert ledger.snapshot().outstanding_reserved == 0

    def test_fail_closed(self) -> None:
        harness = EvalHarness(budget_ledger=_ledger(1))
        with pytest.raises(TrialsLedgerExhausted):
            harness.evaluate_guarded_scorers([BrierScorer()], _records(3))

    def test_release_on_failure(self) -> None:
        ledger = _ledger(10)
        harness = EvalHarness(budget_ledger=ledger)
        with pytest.raises(RuntimeError, match="exploded"):
            harness.evaluate_guarded_scorers([_FailingScorer()], _records(2))
        assert ledger.snapshot().outstanding_reserved == 0

    def test_empty_records_raises(self) -> None:
        harness = EvalHarness(budget_ledger=_ledger(10))
        with pytest.raises(ValueError, match="empty guarded set"):
            harness.evaluate_guarded_scorers([BrierScorer()], [])

    def test_empty_scorers_raises(self) -> None:
        harness = EvalHarness(budget_ledger=_ledger(10))
        with pytest.raises(ValueError, match="at least one scorer"):
            harness.evaluate_guarded_scorers([], _records(2))


class TestHoldoutAccess:
    def test_access_through_governor(self) -> None:
        governor = InMemoryHoldoutGovernor(
            budget=2, source=StaticHoldoutSource(payload={"held": "out"})
        )
        harness = EvalHarness(budget_ledger=_ledger(10), holdout=governor)
        payload = harness.access_holdout(reason="calibration check")
        assert payload == {"held": "out"}

    def test_no_governor_fails_closed(self) -> None:
        harness = EvalHarness(budget_ledger=_ledger(10))
        with pytest.raises(HoldoutUnavailable):
            harness.access_holdout(reason="peek")
