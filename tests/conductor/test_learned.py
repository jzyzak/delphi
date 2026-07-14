"""Tests for the Stage-2 learned-conductor scaffold (Appendix A.1-A.4)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from conductor.corpus import CorpusWriter, InMemoryCorpusStore
from conductor.heuristic import WorkflowTrace
from conductor.learned import (
    GreedyPolicyTrainer,
    LearnedConductorPolicy,
    RLTrainer,
    RolloutDecision,
    TrainingExample,
    anonymize_provenance,
    as_of_safe,
    assert_as_of_safe,
    export_corpus,
    holdout_gated_rollout,
    proper_score_reward,
)
from core.registry.models import (
    EvidenceItem,
    EvidenceSetInput,
    ForecastInput,
    QuestionInput,
    ResolutionInput,
)
from core.registry.store import InMemoryRegistryStore
from evaluation.aggregate import ScoreSummary

_AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _example(route: tuple[str, ...], score: float, *, qid: str = "q") -> TrainingExample:
    return TrainingExample(
        question_id=qid,
        domain="tech",
        features={"n_evidence": 1.0, "revisions": 0.0, "route_len": float(len(route))},
        route=route,
        proper_score=score,
    )


class TestAnonymizeProvenance:
    def test_maps_brands_to_stable_ids(self) -> None:
        prov = {
            "conductor": {"model_version": "claude-zzz"},
            "workers": [{"model_version": "claude-aaa"}, {"model_version": "claude-zzz"}],
        }
        out = anonymize_provenance(prov)
        # Sorted first-appearance: aaa -> model_0, zzz -> model_1.
        assert out["conductor"]["model_version"] == "model_1"
        assert out["workers"][0]["model_version"] == "model_0"
        assert out["workers"][1]["model_version"] == "model_1"

    def test_leaves_non_version_fields_and_tuples(self) -> None:
        prov = {"route": ("researcher", "estimator"), "temp": 0.0}
        assert anonymize_provenance(prov) == {"route": ("researcher", "estimator"), "temp": 0.0}


class TestExportCorpus:
    def _seed(self, *, resolved: bool) -> InMemoryCorpusStore:
        store = InMemoryRegistryStore()
        corpus = InMemoryCorpusStore()
        qid = store.record_question(
            QuestionInput(
                text="Will X ship?",
                question_type="binary",
                domain="tech",
                resolution_criteria="GA.",
            )
        )
        store.record_evidence_set(
            EvidenceSetInput(
                question_id=qid,
                as_of=_AS_OF,
                items=(
                    EvidenceItem(
                        snippet="s", source="src", knowledge_time=datetime(2024, 5, 1, tzinfo=UTC)
                    ),
                ),
            )
        )
        store.record_forecast(
            ForecastInput(
                question_id=qid,
                as_of=_AS_OF,
                probability=0.8,
                rationale="r",
                model_provenance={"conductor": {"model_version": "claude-x"}},
                repro_handle={"as_of": _AS_OF.isoformat()},
            )
        )
        if resolved:
            store.record_resolution(
                ResolutionInput(
                    question_id=qid,
                    resolved_value=1.0,
                    resolved_at=datetime(2025, 1, 1, tzinfo=UTC),
                    source="gov",
                )
            )
        writer = CorpusWriter(store=store, corpus=corpus)
        writer.capture(qid, WorkflowTrace(steps=(), route=("researcher", "estimator"), revisions=1))
        return corpus

    def test_exports_resolved_example_with_features(self) -> None:
        examples = export_corpus(self._seed(resolved=True))
        assert len(examples) == 1
        ex = examples[0]
        assert ex.domain == "tech"
        assert ex.route == ("researcher", "estimator")
        assert ex.features == {"n_evidence": 1.0, "revisions": 1.0, "route_len": 2.0}
        assert ex.proper_score == pytest.approx((0.8 - 1.0) ** 2)
        # Provenance is anonymized (no brand leaks into training).
        assert ex.provenance == {"conductor": {"model_version": "model_0"}}

    def test_skips_unresolved(self) -> None:
        assert export_corpus(self._seed(resolved=False)) == ()


class TestReward:
    def test_positive_when_candidate_better(self) -> None:
        # Lower proper score is better, so a lower candidate earns positive reward.
        assert proper_score_reward(baseline=0.25, candidate=0.10) == pytest.approx(0.15)

    def test_negative_when_candidate_worse(self) -> None:
        assert proper_score_reward(baseline=0.10, candidate=0.25) == pytest.approx(-0.15)

    def test_as_of_safe_true_and_false(self) -> None:
        assert as_of_safe(_AS_OF, [datetime(2024, 5, 1, tzinfo=UTC), _AS_OF])
        assert not as_of_safe(_AS_OF, [datetime(2024, 7, 1, tzinfo=UTC)])

    def test_assert_as_of_safe_raises_on_future_evidence(self) -> None:
        with pytest.raises(ValueError, match="as-of violation"):
            assert_as_of_safe(_AS_OF, [datetime(2024, 7, 1, tzinfo=UTC)])

    def test_assert_as_of_safe_passes_clean(self) -> None:
        assert_as_of_safe(_AS_OF, [datetime(2024, 5, 1, tzinfo=UTC)])  # no raise


class TestTrainer:
    def test_conforms_to_protocol(self) -> None:
        assert isinstance(GreedyPolicyTrainer(), RLTrainer)

    def test_picks_lowest_mean_proper_score_route(self) -> None:
        examples = [
            _example(("a",), 0.30, qid="q1"),
            _example(("a",), 0.10, qid="q2"),  # route a mean = 0.20
            _example(("b",), 0.05, qid="q3"),  # route b mean = 0.05 (best)
        ]
        policy = GreedyPolicyTrainer().train(examples)
        assert isinstance(policy, LearnedConductorPolicy)
        assert policy.route == ("b",)
        assert policy.expected_score == pytest.approx(0.05)
        # Reward = corpus mean (0.15) - best route mean (0.05).
        assert policy.reward == pytest.approx(0.10)
        assert policy.provenance["trainer"] == "greedy_proper_score"
        assert policy.provenance["n_routes"] == 2
        assert policy.emit_route(examples[0]) == ("b",)

    def test_deterministic_tie_break_by_route(self) -> None:
        examples = [_example(("b",), 0.10, qid="q1"), _example(("a",), 0.10, qid="q2")]
        assert GreedyPolicyTrainer().train(examples).route == ("a",)

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            GreedyPolicyTrainer().train([])


def _summary(mean: float) -> ScoreSummary:
    return ScoreSummary(scorer="brier", mean=mean, ci_low=mean - 0.01, ci_high=mean + 0.01, n=50)


class TestRollout:
    def test_promotes_when_learned_beats_heuristic(self) -> None:
        decision = holdout_gated_rollout(
            learned=_summary(0.10),
            heuristic=_summary(0.20),
            calibration_regressed=False,
            leakage_regressed=False,
        )
        assert isinstance(decision, RolloutDecision)
        assert decision.promote
        assert "promote learned" in decision.reason

    def test_keeps_heuristic_when_not_better(self) -> None:
        decision = holdout_gated_rollout(
            learned=_summary(0.20),
            heuristic=_summary(0.20),
            calibration_regressed=False,
            leakage_regressed=False,
        )
        assert not decision.promote
        assert "fine outcome" in decision.reason

    def test_calibration_regression_vetoes(self) -> None:
        decision = holdout_gated_rollout(
            learned=_summary(0.01),
            heuristic=_summary(0.20),
            calibration_regressed=True,
            leakage_regressed=False,
        )
        assert not decision.promote
        assert "calibration regressed" in decision.reason

    def test_leakage_regression_vetoes(self) -> None:
        decision = holdout_gated_rollout(
            learned=_summary(0.01),
            heuristic=_summary(0.20),
            calibration_regressed=False,
            leakage_regressed=True,
        )
        assert not decision.promote
        assert "leakage regressed" in decision.reason
