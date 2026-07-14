"""Store tests for the forecast taxonomy (question/evidence_set/forecast/resolution).

Correctness-critical (CLAUDE.md §2.8): every write/read path, existence check,
and the hash-chain over a mixed question stream are asserted. Deterministic
in-memory backend with an injected clock; no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from core.registry.models import (
    EvidenceItem,
    EvidenceSetInput,
    ForecastInput,
    Quantile,
    QuestionInput,
    ResolutionInput,
)
from core.registry.store import InMemoryRegistryStore, RecordNotFoundError
from tests.registry.conftest import IncrementingClock, make_experiment_input

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)
KT = datetime(2024, 5, 1, tzinfo=UTC)


def make_question(**overrides: Any) -> QuestionInput:
    base: dict[str, Any] = {
        "text": "Will X ship before 2025?",
        "question_type": "binary",
        "domain": "tech",
        "resolution_criteria": "Resolves YES if X is GA before 2025-01-01.",
        "close_time": datetime(2025, 1, 1, tzinfo=UTC),
        "source": "user",
    }
    base.update(overrides)
    return QuestionInput(**base)


def make_forecast(question_id: str, **overrides: Any) -> ForecastInput:
    base: dict[str, Any] = {
        "question_id": question_id,
        "as_of": AS_OF,
        "probability": 0.42,
        "rationale": "base rate + evidence",
        "model_provenance": {"models": ["model-0"], "pipeline_version": "v1"},
        "repro_handle": {"as_of": AS_OF.isoformat(), "cache_ref": "s3://snap/x"},
    }
    base.update(overrides)
    return ForecastInput(**base)


@pytest.fixture
def store() -> InMemoryRegistryStore:
    return InMemoryRegistryStore(clock=IncrementingClock())


class TestQuestion:
    def test_record_and_get_round_trip(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        q = store.get_question(qid)
        assert q.question_id == qid
        assert q.text == "Will X ship before 2025?"
        assert q.domain == "tech"
        assert q.close_time == datetime(2025, 1, 1, tzinfo=UTC)

    def test_get_missing_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.get_question("nope")

    def test_get_question_on_non_question_stream_raises(self, store: InMemoryRegistryStore) -> None:
        exp_id = store.record_experiment(make_experiment_input())
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.get_question(exp_id)

    def test_questions_by_domain_grouped_and_ordered(self, store: InMemoryRegistryStore) -> None:
        q1 = store.record_question(make_question(domain="tech"))
        q2 = store.record_question(make_question(domain="tech"))
        store.record_question(make_question(domain="econ"))
        tech = store.questions_by_domain("tech")
        assert [q.question_id for q in tech] == [q1, q2]  # knowledge-time order
        assert store.questions_by_domain("missing") == ()


class TestEvidenceSet:
    def test_record_and_list(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        items = (EvidenceItem(snippet="s", source="src", knowledge_time=KT),)
        evs_id = store.record_evidence_set(
            EvidenceSetInput(question_id=qid, as_of=AS_OF, items=items)
        )
        sets = store.evidence_sets_for(qid)
        assert len(sets) == 1
        assert sets[0].evidence_set_id == evs_id
        assert sets[0].items[0].snippet == "s"
        assert sets[0].items[0].knowledge_time == KT

    def test_empty_list(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        assert store.evidence_sets_for(qid) == ()

    def test_unknown_question_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.record_evidence_set(EvidenceSetInput(question_id="nope", as_of=AS_OF))

    def test_evidence_sets_for_unknown_question_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.evidence_sets_for("nope")


class TestForecast:
    def test_record_binary_and_get(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        fc_id = store.record_forecast(make_forecast(qid))
        fc = store.get_forecast(fc_id)
        assert fc.forecast_id == fc_id
        assert fc.probability == pytest.approx(0.42)
        assert store.forecasts_for(qid)[0].forecast_id == fc_id

    def test_record_distribution(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question(question_type="numeric"))
        fc_id = store.record_forecast(
            make_forecast(
                qid,
                probability=None,
                quantiles=(
                    Quantile(level=0.1, value=1.0),
                    Quantile(level=0.9, value=5.0),
                ),
            )
        )
        fc = store.get_forecast(fc_id)
        assert fc.probability is None
        assert [q.value for q in (fc.quantiles or ())] == [1.0, 5.0]

    def test_forecast_links_valid_evidence_set(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        evs_id = store.record_evidence_set(EvidenceSetInput(question_id=qid, as_of=AS_OF))
        fc_id = store.record_forecast(make_forecast(qid, evidence_set_id=evs_id))
        assert store.get_forecast(fc_id).evidence_set_id == evs_id

    def test_forecast_with_unknown_evidence_set_raises(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        with pytest.raises(RecordNotFoundError, match="No evidence set"):
            store.record_forecast(make_forecast(qid, evidence_set_id="evs_missing"))

    def test_forecast_unknown_question_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.record_forecast(make_forecast("nope"))

    def test_get_forecast_missing_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No forecast"):
            store.get_forecast("fc_missing")

    def test_get_forecast_scans_past_non_matching(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        store.record_forecast(make_forecast(qid, probability=0.1))
        second = store.record_forecast(make_forecast(qid, probability=0.9))
        # Fetching the second forecast forces the loop past the first (non-match).
        assert store.get_forecast(second).probability == pytest.approx(0.9)
        # Missing id while forecasts exist also exercises the no-match path fully.
        with pytest.raises(RecordNotFoundError, match="No forecast"):
            store.get_forecast("fc_absent")

    def test_forecasts_for_unknown_question_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.forecasts_for("nope")


class TestResolution:
    def test_record_and_list(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        rid = store.record_resolution(
            ResolutionInput(
                question_id=qid, resolved_value=1.0, resolved_at=AS_OF, source="official"
            )
        )
        res = store.resolutions_for(qid)
        assert len(res) == 1
        assert res[0].resolution_id == rid
        assert res[0].resolved_value == pytest.approx(1.0)

    def test_resolution_links_valid_forecast(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        fc_id = store.record_forecast(make_forecast(qid))
        rid = store.record_resolution(
            ResolutionInput(
                question_id=qid,
                forecast_id=fc_id,
                resolved_value=0.0,
                resolved_at=AS_OF,
                source="official",
            )
        )
        linked = store.resolutions_for_forecast(fc_id)
        assert [r.resolution_id for r in linked] == [rid]

    def test_resolution_with_unknown_forecast_raises(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        with pytest.raises(RecordNotFoundError, match="No forecast"):
            store.record_resolution(
                ResolutionInput(
                    question_id=qid,
                    forecast_id="fc_missing",
                    resolved_value=1.0,
                    resolved_at=AS_OF,
                    source="official",
                )
            )

    def test_resolution_unknown_question_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.record_resolution(
                ResolutionInput(
                    question_id="nope", resolved_value=1.0, resolved_at=AS_OF, source="s"
                )
            )

    def test_resolutions_for_unknown_question_raises(self, store: InMemoryRegistryStore) -> None:
        with pytest.raises(RecordNotFoundError, match="No question"):
            store.resolutions_for("nope")

    def test_resolutions_for_forecast_empty(self, store: InMemoryRegistryStore) -> None:
        assert store.resolutions_for_forecast("fc_none") == ()


class TestChainIntegrityOverQuestionStream:
    def test_full_lifecycle_chain_verifies(self, store: InMemoryRegistryStore) -> None:
        qid = store.record_question(make_question())
        evs_id = store.record_evidence_set(
            EvidenceSetInput(
                question_id=qid,
                as_of=AS_OF,
                items=(EvidenceItem(snippet="s", source="src", knowledge_time=KT),),
            )
        )
        fc_id = store.record_forecast(make_forecast(qid, evidence_set_id=evs_id))
        store.record_resolution(
            ResolutionInput(
                question_id=qid,
                forecast_id=fc_id,
                resolved_value=1.0,
                resolved_at=AS_OF,
                source="official",
            )
        )
        verification = store.verify_chain(qid)
        assert verification.ok
        # question genesis + evidence_set + forecast + resolution == 4 records.
        assert len(store._stream_events(qid)) == 4  # noqa: SLF001
