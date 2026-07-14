"""Validation tests for the forecast taxonomy models (DELPHI §3/§5).

Correctness-critical (CLAUDE.md §2.8): every validator branch and error path is
asserted. Deterministic; no network, no wall clock.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.registry.models import (
    EvidenceItem,
    EvidenceSet,
    EvidenceSetInput,
    Forecast,
    ForecastInput,
    Quantile,
    Question,
    QuestionInput,
    Resolution,
    ResolutionInput,
)

AS_OF = datetime(2024, 6, 1, tzinfo=UTC)
KT = datetime(2024, 5, 1, tzinfo=UTC)
NAIVE = datetime(2024, 6, 1)  # noqa: DTZ001


def _provenance() -> dict[str, object]:
    return {"models": ["model-0"], "pipeline_version": "v1"}


def _repro() -> dict[str, object]:
    return {"as_of": AS_OF.isoformat(), "cache_ref": "s3://snap/abc"}


class TestQuestionInput:
    def test_valid(self) -> None:
        q = QuestionInput(
            text="Will X ship by 2025?",
            question_type="binary",
            domain="tech",
            resolution_criteria="Resolves YES if X is generally available before 2025-12-31.",
            close_time=AS_OF,
        )
        assert q.question_type == "binary"
        assert q.close_time == AS_OF

    @pytest.mark.parametrize("field", ["text", "domain", "resolution_criteria"])
    def test_rejects_blank_required_fields(self, field: str) -> None:
        base = {
            "text": "Q?",
            "question_type": "binary",
            "domain": "tech",
            "resolution_criteria": "criteria",
        }
        base[field] = "  "
        with pytest.raises(ValueError, match="required"):
            QuestionInput(**base)  # type: ignore[arg-type]

    def test_close_time_none_allowed(self) -> None:
        q = QuestionInput(
            text="Q?", question_type="numeric", domain="econ", resolution_criteria="c"
        )
        assert q.close_time is None

    def test_rejects_naive_close_time(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            QuestionInput(
                text="Q?",
                question_type="binary",
                domain="tech",
                resolution_criteria="c",
                close_time=NAIVE,
            )


class TestEvidenceItem:
    def test_valid_with_defaults(self) -> None:
        item = EvidenceItem(snippet="s", source="src", knowledge_time=KT)
        assert item.score == 0.0
        assert item.source_id == ""

    @pytest.mark.parametrize("field", ["snippet", "source"])
    def test_rejects_blank(self, field: str) -> None:
        base = {"snippet": "s", "source": "src", "knowledge_time": KT}
        base[field] = " "
        with pytest.raises(ValueError, match="required"):
            EvidenceItem(**base)  # type: ignore[arg-type]

    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            EvidenceItem(snippet="s", source="src", knowledge_time=KT, score=-0.1)

    def test_rejects_naive_knowledge_time(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            EvidenceItem(snippet="s", source="src", knowledge_time=NAIVE)


class TestEvidenceSetInput:
    def test_valid_empty_items(self) -> None:
        es = EvidenceSetInput(question_id="q1", as_of=AS_OF)
        assert es.items == ()

    def test_rejects_blank_question_id(self) -> None:
        with pytest.raises(ValueError, match="question_id is required"):
            EvidenceSetInput(question_id=" ", as_of=AS_OF)

    def test_rejects_naive_as_of(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            EvidenceSetInput(question_id="q1", as_of=NAIVE)

    def test_item_within_as_of_ok(self) -> None:
        es = EvidenceSetInput(
            question_id="q1",
            as_of=AS_OF,
            items=(EvidenceItem(snippet="s", source="src", knowledge_time=KT),),
        )
        assert len(es.items) == 1

    def test_rejects_post_as_of_item(self) -> None:
        future = datetime(2024, 7, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="no look-ahead"):
            EvidenceSetInput(
                question_id="q1",
                as_of=AS_OF,
                items=(EvidenceItem(snippet="s", source="src", knowledge_time=future),),
            )


class TestQuantile:
    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range_level(self, level: float) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Quantile(level=level, value=1.0)

    def test_valid(self) -> None:
        assert Quantile(level=0.5, value=3.0).value == 3.0


class TestForecastInput:
    def _binary(self, **overrides: object) -> ForecastInput:
        base: dict[str, object] = {
            "question_id": "q1",
            "as_of": AS_OF,
            "probability": 0.4,
            "rationale": "because evidence",
            "model_provenance": _provenance(),
            "repro_handle": _repro(),
        }
        base.update(overrides)
        return ForecastInput(**base)  # type: ignore[arg-type]

    def test_valid_binary(self) -> None:
        fc = self._binary()
        assert fc.probability == pytest.approx(0.4)
        assert fc.quantiles is None

    def test_valid_distribution(self) -> None:
        fc = ForecastInput(
            question_id="q1",
            as_of=AS_OF,
            quantiles=(
                Quantile(level=0.1, value=1.0),
                Quantile(level=0.5, value=2.0),
                Quantile(level=0.9, value=2.0),
            ),
            rationale="r",
            model_provenance=_provenance(),
            repro_handle=_repro(),
        )
        assert fc.probability is None
        assert len(fc.quantiles or ()) == 3

    def test_rejects_both_probability_and_quantiles(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            self._binary(quantiles=(Quantile(level=0.5, value=1.0),))

    def test_rejects_neither(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            self._binary(probability=None)

    def test_rejects_empty_quantiles(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            self._binary(probability=None, quantiles=())

    def test_rejects_probability_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1"):
            self._binary(probability=1.5)

    def test_rejects_non_increasing_levels(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            self._binary(
                probability=None,
                quantiles=(
                    Quantile(level=0.5, value=1.0),
                    Quantile(level=0.5, value=2.0),
                ),
            )

    def test_rejects_decreasing_values(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            self._binary(
                probability=None,
                quantiles=(
                    Quantile(level=0.4, value=5.0),
                    Quantile(level=0.6, value=2.0),
                ),
            )

    def test_rejects_empty_provenance(self) -> None:
        with pytest.raises(ValueError, match="model_provenance is required"):
            self._binary(model_provenance={})

    def test_rejects_empty_repro_handle(self) -> None:
        with pytest.raises(ValueError, match="repro_handle is required"):
            self._binary(repro_handle={})

    @pytest.mark.parametrize("field", ["question_id", "rationale"])
    def test_rejects_blank_strings(self, field: str) -> None:
        with pytest.raises(ValueError, match="required"):
            self._binary(**{field: " "})

    def test_rejects_naive_as_of(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            self._binary(as_of=NAIVE)


class TestResolutionInput:
    def test_valid(self) -> None:
        r = ResolutionInput(
            question_id="q1", resolved_value=1.0, resolved_at=AS_OF, source="official"
        )
        assert r.resolved_value == pytest.approx(1.0)
        assert r.forecast_id is None

    @pytest.mark.parametrize("field", ["question_id", "source"])
    def test_rejects_blank(self, field: str) -> None:
        base = {
            "question_id": "q1",
            "resolved_value": 0.0,
            "resolved_at": AS_OF,
            "source": "official",
        }
        base[field] = " "
        with pytest.raises(ValueError, match="required"):
            ResolutionInput(**base)  # type: ignore[arg-type]

    def test_rejects_naive_resolved_at(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            ResolutionInput(
                question_id="q1", resolved_value=0.0, resolved_at=NAIVE, source="official"
            )


class TestStoredRecordsRejectNaiveKnowledgeTime:
    def test_question(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            Question(
                text="Q?",
                question_type="binary",
                domain="tech",
                resolution_criteria="c",
                question_id="q1",
                knowledge_time=NAIVE,
            )

    def test_evidence_set(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            EvidenceSet(question_id="q1", as_of=AS_OF, evidence_set_id="evs1", knowledge_time=NAIVE)

    def test_forecast(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            Forecast(
                question_id="q1",
                as_of=AS_OF,
                probability=0.5,
                rationale="r",
                model_provenance=_provenance(),
                repro_handle=_repro(),
                forecast_id="fc1",
                knowledge_time=NAIVE,
            )

    def test_resolution(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            Resolution(
                question_id="q1",
                resolved_value=1.0,
                resolved_at=AS_OF,
                source="official",
                resolution_id="rsl1",
                knowledge_time=NAIVE,
            )
