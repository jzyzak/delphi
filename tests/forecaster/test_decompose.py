"""Unit tests for the decomposition stage (C4.2)."""

from __future__ import annotations

import pytest

from forecaster.stages.decompose import decompose_question, recompose
from intake.llm import FixtureStructuredLLM


def test_decomposes_with_rule() -> None:
    llm = FixtureStructuredLLM({"sub_questions": ["Will A?", "Will B?", ""], "rule": "product"})
    d = decompose_question("Will A and B?", llm=llm)
    assert [s.text for s in d.sub_questions] == ["Will A?", "Will B?"]
    assert d.rule == "product"
    assert d.provenance["n_sub_questions"] == 2


def test_unknown_rule_becomes_none() -> None:
    d = decompose_question("Q?", llm=FixtureStructuredLLM({"rule": "magic"}))
    assert d.rule == "none"
    assert d.sub_questions == ()


def test_missing_payload_defaults() -> None:
    d = decompose_question("Q?", llm=FixtureStructuredLLM({}))
    assert d.rule == "none"
    assert d.sub_questions == ()


def test_non_list_sub_questions_ignored() -> None:
    d = decompose_question(
        "Q?", llm=FixtureStructuredLLM({"sub_questions": "nope", "rule": "product"})
    )
    assert d.sub_questions == ()
    assert d.rule == "product"


class TestRecompose:
    def test_product(self) -> None:
        assert recompose("product", [0.5, 0.5]) == pytest.approx(0.25)

    def test_scenario_tree_sums_and_clamps(self) -> None:
        assert recompose("scenario_tree", [0.4, 0.3]) == pytest.approx(0.7)
        assert recompose("scenario_tree", [0.8, 0.8]) == 1.0

    def test_none_passes_through_first(self) -> None:
        assert recompose("none", [0.42, 0.9]) == pytest.approx(0.42)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            recompose("product", [])

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="probabilities in"):
            recompose("product", [1.5])
