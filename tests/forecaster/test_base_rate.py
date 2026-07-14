"""Unit tests for the reference-class / base-rate stage (C4.1)."""

from __future__ import annotations

from datetime import UTC, datetime

from core.forecast.search import Evidence
from forecaster.stages.base_rate import estimate_base_rate
from intake.llm import FixtureStructuredLLM

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)


def _ev(source_id: str) -> Evidence:
    return Evidence(
        snippet=f"snippet for {source_id}",
        source="hosted",
        source_id=source_id,
        knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
        score=0.5,
    )


def test_extracts_prior_and_binds_citations() -> None:
    llm = FixtureStructuredLLM(
        {
            "reference_class": "annual elections",
            "base_rate": 0.3,
            "rationale": "history says so",
            "citations": ["http://a", "http://unknown"],
        }
    )
    est = estimate_base_rate(
        "Will X win?", [_ev("http://a"), _ev("http://b")], llm=llm, as_of=AS_OF
    )
    assert est.prior == 0.3
    assert est.reference_class == "annual elections"
    assert est.citations == ("http://a",)  # unknown id dropped
    assert est.provenance["defaulted_prior"] is False


def test_defaults_prior_when_missing() -> None:
    est = estimate_base_rate("Q?", [_ev("http://a")], llm=FixtureStructuredLLM({}), as_of=AS_OF)
    assert est.prior == 0.5
    assert est.provenance["defaulted_prior"] is True
    assert est.citations == ("http://a",)  # falls back to all evidence ids


def test_clamps_extreme_rates() -> None:
    lo = estimate_base_rate("Q?", [], llm=FixtureStructuredLLM({"base_rate": 0.0}), as_of=AS_OF)
    hi = estimate_base_rate("Q?", [], llm=FixtureStructuredLLM({"base_rate": 1.0}), as_of=AS_OF)
    assert 0.0 < lo.prior < 0.5
    assert 0.5 < hi.prior < 1.0


def test_non_numeric_rate_defaults() -> None:
    est = estimate_base_rate("Q?", [], llm=FixtureStructuredLLM({"base_rate": "high"}), as_of=AS_OF)
    assert est.prior == 0.5
    assert est.provenance["defaulted_prior"] is True


def test_invalid_citations_type_falls_back() -> None:
    llm = FixtureStructuredLLM({"base_rate": 0.4, "citations": "not-a-list"})
    est = estimate_base_rate("Q?", [_ev("http://b"), _ev("http://a")], llm=llm, as_of=AS_OF)
    assert est.citations == ("http://a", "http://b")  # sorted evidence ids


def test_no_evidence_yields_empty_citations() -> None:
    est = estimate_base_rate("Q?", [], llm=FixtureStructuredLLM({"base_rate": 0.4}), as_of=AS_OF)
    assert est.citations == ()
    assert est.reference_class == "unspecified reference class"
