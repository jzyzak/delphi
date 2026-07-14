"""Unit + leakage + determinism tests for the generic as-of evidence view.

Correctness-critical module (CLAUDE.md §2.8): 100% coverage, deterministic and
hermetic (no network, explicit as-of, no wall clock).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from core.pit.adapters.fixtures import utc_dt
from core.pit.models import FactRecord
from core.pit.store import InMemoryPitStore
from core.pit.view import (
    EvidenceQuery,
    EvidenceRecord,
    LeakageError,
    PitEvidenceView,
)

DATASET = "news"
OTHER_DATASET = "filings"
EMPTY_DATASET = "empty_payloads"

T_EFF = utc_dt(2024, 1, 1)
KT_EARLY = utc_dt(2024, 1, 2)
KT_MID = utc_dt(2024, 1, 5)
KT_LATE = utc_dt(2024, 6, 1)
AS_OF = utc_dt(2024, 1, 10)


def _fact(
    *,
    dataset: str = DATASET,
    entity_id: str = "A",
    effective_time: datetime = T_EFF,
    knowledge_time: datetime = KT_EARLY,
    values: dict[str, object] | None = None,
) -> FactRecord:
    return FactRecord(
        dataset=dataset,
        entity_id=entity_id,
        effective_time=effective_time,
        knowledge_time=knowledge_time,
        values={"snippet": "x"} if values is None else values,
    )


@pytest.fixture
def seeded_view() -> PitEvidenceView:
    store = InMemoryPitStore()
    store.append(
        [
            _fact(entity_id="A", knowledge_time=KT_MID, values={"snippet": "a-mid"}),
            _fact(entity_id="B", knowledge_time=KT_EARLY, values={"snippet": "b-early"}),
            # Post-as-of: must never surface.
            _fact(entity_id="C", knowledge_time=KT_LATE, values={"snippet": "c-late"}),
            _fact(dataset=OTHER_DATASET, entity_id="A", knowledge_time=KT_EARLY),
        ]
    )
    return PitEvidenceView(store)


class TestEvidenceAsOf:
    def test_returns_only_facts_known_as_of(self, seeded_view: PitEvidenceView) -> None:
        result = seeded_view.evidence_as_of(
            EvidenceQuery(datasets=(DATASET, OTHER_DATASET), as_of=AS_OF)
        )
        # The KT_LATE (entity C) fact is filtered out structurally.
        assert all(r.knowledge_time <= AS_OF for r in result)
        assert {(r.dataset, r.entity_id) for r in result} == {
            (DATASET, "A"),
            (DATASET, "B"),
            (OTHER_DATASET, "A"),
        }

    def test_no_lookahead_when_as_of_precedes_all_late_facts(
        self, seeded_view: PitEvidenceView
    ) -> None:
        result = seeded_view.evidence_as_of(EvidenceQuery(datasets=(DATASET,), as_of=KT_EARLY))
        assert [r.entity_id for r in result] == ["B"]

    def test_deterministic_ordering(self, seeded_view: PitEvidenceView) -> None:
        result = seeded_view.evidence_as_of(
            EvidenceQuery(datasets=(DATASET, OTHER_DATASET), as_of=AS_OF)
        )
        keys = [(r.knowledge_time, r.dataset, r.entity_id, r.effective_time) for r in result]
        assert keys == sorted(keys)
        # B(early) and filings/A(early) share KT; ordering breaks ties by dataset.
        assert result[0].knowledge_time == KT_EARLY
        assert result[0].dataset == OTHER_DATASET  # "filings" < "news"

    def test_entity_filter(self, seeded_view: PitEvidenceView) -> None:
        result = seeded_view.evidence_as_of(
            EvidenceQuery(datasets=(DATASET,), as_of=AS_OF, entity_ids=("B",))
        )
        assert [r.entity_id for r in result] == ["B"]

    def test_limit_keeps_first_in_order(self, seeded_view: PitEvidenceView) -> None:
        result = seeded_view.evidence_as_of(
            EvidenceQuery(datasets=(DATASET, OTHER_DATASET), as_of=AS_OF, limit=1)
        )
        assert len(result) == 1
        assert result[0].knowledge_time == KT_EARLY

    def test_effective_range_filters(self) -> None:
        kt = utc_dt(2024, 3, 15)
        store = InMemoryPitStore()
        store.append(
            [
                _fact(entity_id="A", effective_time=utc_dt(2024, 1, 1), knowledge_time=kt),
                _fact(entity_id="A", effective_time=utc_dt(2024, 3, 1), knowledge_time=kt),
            ]
        )
        view = PitEvidenceView(store)
        result = view.evidence_as_of(
            EvidenceQuery(
                datasets=(DATASET,),
                as_of=utc_dt(2024, 4, 1),
                effective_range=(utc_dt(2024, 1, 1), utc_dt(2024, 1, 31)),
            )
        )
        assert [r.effective_time for r in result] == [utc_dt(2024, 1, 1)]

    def test_empty_result(self) -> None:
        view = PitEvidenceView(InMemoryPitStore())
        assert view.evidence_as_of(EvidenceQuery(datasets=(DATASET,), as_of=AS_OF)) == ()

    def test_heterogeneous_payloads_drop_null_fills(self) -> None:
        store = InMemoryPitStore()
        store.append(
            [
                _fact(entity_id="A", knowledge_time=KT_EARLY, values={"headline": "h"}),
                _fact(entity_id="B", knowledge_time=KT_EARLY, values={"body": "b"}),
            ]
        )
        view = PitEvidenceView(store)
        result = view.evidence_as_of(EvidenceQuery(datasets=(DATASET,), as_of=AS_OF))
        by_entity = {r.entity_id: r.values for r in result}
        assert by_entity == {"A": {"headline": "h"}, "B": {"body": "b"}}

    def test_empty_payload_struct(self) -> None:
        store = InMemoryPitStore()
        store.append([_fact(dataset=EMPTY_DATASET, entity_id="A", values={})])
        view = PitEvidenceView(store)
        result = view.evidence_as_of(EvidenceQuery(datasets=(EMPTY_DATASET,), as_of=AS_OF))
        assert result[0].values == {}


class _LeakyStore(InMemoryPitStore):
    """Broken backend that ignores the as-of ceiling (contract violation)."""

    def corpus_as_of(
        self,
        *,
        datasets: object,
        as_of: object,
        effective_range: object = None,
    ) -> pl.DataFrame:
        return super().corpus_as_of(
            datasets=datasets,  # type: ignore[arg-type]
            as_of=utc_dt(2999, 1, 1),
            effective_range=effective_range,  # type: ignore[arg-type]
        )


class TestLeakageGuard:
    def test_post_as_of_row_raises_leakage_error(self) -> None:
        store = _LeakyStore()
        store.append([_fact(entity_id="C", knowledge_time=KT_LATE)])
        view = PitEvidenceView(store)
        with pytest.raises(LeakageError, match="Leakage"):
            view.evidence_as_of(EvidenceQuery(datasets=(DATASET,), as_of=AS_OF))


class TestEvidenceQueryValidation:
    def test_rejects_empty_datasets(self) -> None:
        with pytest.raises(ValueError, match="datasets must be non-empty"):
            EvidenceQuery(datasets=(), as_of=AS_OF)

    def test_rejects_reversed_effective_range(self) -> None:
        with pytest.raises(ValueError, match="start must be <= end"):
            EvidenceQuery(
                datasets=(DATASET,),
                as_of=AS_OF,
                effective_range=(utc_dt(2024, 2, 1), utc_dt(2024, 1, 1)),
            )

    def test_rejects_naive_as_of(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            EvidenceQuery(datasets=(DATASET,), as_of=datetime(2024, 1, 10))  # noqa: DTZ001

    def test_none_effective_range_allowed(self) -> None:
        query = EvidenceQuery(datasets=(DATASET,), as_of=AS_OF, effective_range=None)
        assert query.effective_range is None

    def test_rejects_zero_limit(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            EvidenceQuery(datasets=(DATASET,), as_of=AS_OF, limit=0)


class TestEvidenceRecordValidation:
    def test_rejects_naive_knowledge_time(self) -> None:
        with pytest.raises(ValueError, match="Naive datetimes"):
            EvidenceRecord(
                dataset=DATASET,
                entity_id="A",
                effective_time=T_EFF,
                knowledge_time=datetime(2024, 1, 2),  # noqa: DTZ001
                values={},
            )
