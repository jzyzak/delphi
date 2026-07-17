"""Integration tests for PostgresRegistryStore (skipped when DELPHI_TEST_PG_DSN unset).

These exercise the real DB-level append-only enforcement (R1) and a round-trip
that mirrors the in-memory reference behavior — against the dedicated TEST
database only (see tests/conftest.py::postgres_test_dsn).
"""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from core.registry.models import (
    DecisionInput,
    EvidenceSetInput,
    ForecastInput,
    QuestionInput,
    ResultInput,
)
from core.registry.store import PostgresRegistryStore
from tests.conftest import postgres_test_dsn
from tests.registry.conftest import make_experiment_input

pytestmark = pytest.mark.postgres


def test_forecast_persists_across_a_separate_connection(
    postgres_store: PostgresRegistryStore,
) -> None:
    """Regression: writes preceded by a read (record_forecast reads the question
    first) must COMMIT durably. With autocommit off, ``conn.transaction()`` was
    demoted to a savepoint and the forecast was silently dropped on close — a §3
    "no silent forecasts" violation. A separate connection can only see the row
    if it was truly committed."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    qid = postgres_store.record_question(
        QuestionInput(
            text="Will the fix hold?",
            question_type="binary",
            domain="test",
            resolution_criteria="x",
            close_time=now,
            source="regression",
        )
    )
    evs = postgres_store.record_evidence_set(EvidenceSetInput(question_id=qid, as_of=now, items=()))
    fid = postgres_store.record_forecast(
        ForecastInput(
            question_id=qid,
            as_of=now,
            probability=0.5,
            quantiles=None,
            rationale="r",
            evidence_set_id=evs,
            model_provenance={"m": "x"},
            trace={},
            calibration_metadata={},
            uncertainty=None,
            repro_handle={"as_of": now.isoformat()},
        )
    )

    other = PostgresRegistryStore.connect(postgres_test_dsn(), migrate=False)
    try:
        assert len(other.evidence_sets_for(qid)) == 1
        assert len(other.forecasts_for(qid)) == 1
        assert other.get_forecast(fid).probability == 0.5
    finally:
        other.close()


def test_round_trip_and_chain(postgres_store: PostgresRegistryStore) -> None:
    exp_id = postgres_store.record_experiment(make_experiment_input())
    postgres_store.record_result(
        ResultInput(experiment_id=exp_id, status="success", metrics={"sharpe": 1.2})
    )
    postgres_store.record_decision(
        DecisionInput(
            experiment_id=exp_id,
            outcome="promote",
            deciding_component="gates.v1",
            component_version="1.0.0",
            rationale="passes",
        )
    )
    fetched = postgres_store.get_experiment(exp_id)
    assert fetched.experiment_id == exp_id
    assert postgres_store.verify_chain(exp_id).ok
    assert postgres_store.results_for(exp_id)[0].metrics["sharpe"] == 1.2


def test_db_rejects_update(postgres_store: PostgresRegistryStore) -> None:
    postgres_store.record_experiment(make_experiment_input())
    with (
        pytest.raises(psycopg.errors.RaiseException),  # type: ignore[attr-defined]
        postgres_store._conn.cursor() as cur,  # noqa: SLF001
    ):
        cur.execute("UPDATE registry_events SET payload = '{}'::jsonb")
    postgres_store._conn.rollback()  # noqa: SLF001


def test_db_rejects_delete(postgres_store: PostgresRegistryStore) -> None:
    postgres_store.record_experiment(make_experiment_input())
    with (
        pytest.raises(psycopg.errors.RaiseException),  # type: ignore[attr-defined]
        postgres_store._conn.cursor() as cur,  # noqa: SLF001
    ):
        cur.execute("DELETE FROM registry_events")
    postgres_store._conn.rollback()  # noqa: SLF001


def test_many_independent_streams(postgres_store: PostgresRegistryStore) -> None:
    # A single psycopg connection is not thread-safe, so this asserts correctness
    # across many independent streams; per-stream advisory locking is exercised by
    # _persist on every append.
    ids = [
        postgres_store.record_experiment(make_experiment_input(niche=f"n{i}")) for i in range(10)
    ]
    assert len(set(ids)) == 10
    for exp_id in ids:
        assert postgres_store.verify_chain(exp_id).ok
