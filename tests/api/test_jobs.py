"""Unit tests for the async forecast-job store + manager (api/jobs.py).

Hermetic and deterministic (§2.8): frozen/steppable clocks, injected sleepers,
and synchronous or deferred executors — no real time, threads only in the one
integration test that exercises the default ThreadPoolExecutor path.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import BaseModel

from api.compliance import ProviderOptOutError
from api.jobs import (
    MAX_WAIT_S,
    ForecastJob,
    InMemoryJobStore,
    JobManager,
    JobStatus,
)

T0 = datetime(2024, 6, 1, tzinfo=UTC)
PAYLOAD = {"question": "Will X ship?", "as_of": "2024-06-01T00:00:00+00:00", "tier": "delphi"}


class _StubResult(BaseModel):
    probability: float = 0.6


class Timeline:
    """Steppable fake time: ``clock`` reads it, ``sleep`` advances it."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


class InlineExecutor:
    """Runs submissions synchronously (deterministic completion)."""

    def __init__(self) -> None:
        self.submissions = 0

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> None:
        self.submissions += 1
        fn(*args)

    def shutdown(self, wait: bool = True) -> None:  # pragma: no cover - not exercised
        del wait


class DeferredExecutor:
    """Collects submissions to run manually (jobs observable while queued)."""

    def __init__(self) -> None:
        self.pending: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> None:
        self.pending.append((fn, args))

    def run_all(self) -> None:
        pending, self.pending = self.pending, []
        for fn, args in pending:
            fn(*args)

    def shutdown(self, wait: bool = True) -> None:  # pragma: no cover - not exercised
        del wait


def _job(job_id: str = "fj-1", **overrides: Any) -> ForecastJob:
    defaults: dict[str, Any] = {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "request": dict(PAYLOAD),
        "created_at": T0,
    }
    defaults.update(overrides)
    return ForecastJob(**defaults)


class TestInMemoryJobStore:
    def test_create_and_get_round_trip(self) -> None:
        store = InMemoryJobStore()
        job, created = store.create(_job())
        assert created is True
        assert store.get("fj-1") == job
        assert len(store) == 1

    def test_get_unknown_returns_none(self) -> None:
        assert InMemoryJobStore().get("nope") is None

    def test_duplicate_idempotency_key_returns_existing(self) -> None:
        store = InMemoryJobStore()
        first, _ = store.create(_job("fj-1", idempotency_key="k"))
        second, created = store.create(_job("fj-2", idempotency_key="k"))
        assert created is False
        assert second.job_id == first.job_id
        assert len(store) == 1

    def test_keyless_jobs_never_collide(self) -> None:
        store = InMemoryJobStore()
        _, created_a = store.create(_job("fj-1"))
        _, created_b = store.create(_job("fj-2"))
        assert created_a and created_b
        assert len(store) == 2

    def test_claim_transitions_queued_to_running(self) -> None:
        store = InMemoryJobStore()
        store.create(_job())
        claimed = store.claim("fj-1", started_at=T0)
        assert claimed is not None
        assert claimed.status is JobStatus.RUNNING
        assert claimed.started_at == T0

    def test_claim_is_at_most_once(self) -> None:
        store = InMemoryJobStore()
        store.create(_job())
        assert store.claim("fj-1", started_at=T0) is not None
        assert store.claim("fj-1", started_at=T0) is None

    def test_claim_unknown_returns_none(self) -> None:
        assert InMemoryJobStore().claim("nope", started_at=T0) is None

    def test_complete_requires_running(self) -> None:
        store = InMemoryJobStore()
        store.create(_job())
        store.complete("fj-1", result={"ok": True}, finished_at=T0)  # still queued: no-op
        job = store.get("fj-1")
        assert job is not None and job.status is JobStatus.QUEUED
        store.claim("fj-1", started_at=T0)
        store.complete("fj-1", result={"ok": True}, finished_at=T0)
        job = store.get("fj-1")
        assert job is not None
        assert job.status is JobStatus.SUCCEEDED
        assert job.result == {"ok": True}
        assert job.finished_at == T0

    def test_complete_unknown_is_noop(self) -> None:
        InMemoryJobStore().complete("nope", result={}, finished_at=T0)

    def test_fail_from_queued_and_running(self) -> None:
        store = InMemoryJobStore()
        store.create(_job("fj-q"))
        store.fail("fj-q", error="boom", finished_at=T0)
        queued = store.get("fj-q")
        assert queued is not None and queued.status is JobStatus.FAILED
        assert queued.error == "boom"

        store.create(_job("fj-r"))
        store.claim("fj-r", started_at=T0)
        store.fail("fj-r", error="boom", finished_at=T0)
        running = store.get("fj-r")
        assert running is not None and running.status is JobStatus.FAILED

    def test_terminal_jobs_are_immutable(self) -> None:
        store = InMemoryJobStore()
        store.create(_job())
        store.claim("fj-1", started_at=T0)
        store.complete("fj-1", result={"ok": True}, finished_at=T0)
        store.fail("fj-1", error="late", finished_at=T0)  # no-op on succeeded
        job = store.get("fj-1")
        assert job is not None
        assert job.status is JobStatus.SUCCEEDED
        assert job.error is None

    def test_fail_unknown_is_noop(self) -> None:
        InMemoryJobStore().fail("nope", error="x", finished_at=T0)


def _manager(
    *,
    store: InMemoryJobStore | None = None,
    runner: Callable[[Any], BaseModel] | None = None,
    executor: Any | None = None,
    timeline: Timeline | None = None,
    stale_after_s: float = 1800.0,
) -> tuple[JobManager, InMemoryJobStore, Timeline]:
    store = store or InMemoryJobStore()
    timeline = timeline or Timeline()
    ids = iter(f"fj-{i}" for i in range(1, 100))
    manager = JobManager(
        store=store,
        runner=runner or (lambda payload: _StubResult()),
        executor=executor if executor is not None else InlineExecutor(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        id_factory=lambda: next(ids),
        stale_after_s=stale_after_s,
    )
    return manager, store, timeline


class TestJobManagerSubmit:
    def test_submit_executes_and_returns_refreshed_job(self) -> None:
        calls: list[Any] = []

        def runner(payload: Any) -> BaseModel:
            calls.append(payload)
            return _StubResult(probability=0.42)

        manager, store, _ = _manager(runner=runner)
        job, created = manager.submit(PAYLOAD)
        assert created is True
        assert job.status is JobStatus.SUCCEEDED
        assert job.result == {"probability": 0.42}
        assert calls == [PAYLOAD]
        persisted = store.get(job.job_id)
        assert persisted is not None and persisted.status is JobStatus.SUCCEEDED

    def test_idempotent_resubmit_runs_once(self) -> None:
        calls: list[Any] = []

        def runner(payload: Any) -> BaseModel:
            calls.append(payload)
            return _StubResult()

        manager, _, _ = _manager(runner=runner)
        first, created_first = manager.submit(PAYLOAD, idempotency_key="k")
        second, created_second = manager.submit(PAYLOAD, idempotency_key="k")
        assert created_first is True
        assert created_second is False
        assert second.job_id == first.job_id
        assert len(calls) == 1

    def test_distinct_keys_create_distinct_jobs(self) -> None:
        manager, store, _ = _manager()
        a, _ = manager.submit(PAYLOAD, idempotency_key="a")
        b, _ = manager.submit(PAYLOAD, idempotency_key="b")
        assert a.job_id != b.job_id
        assert len(store) == 2

    def test_provider_opt_out_fails_job(self) -> None:
        def runner(payload: Any) -> BaseModel:
            raise ProviderOptOutError("all providers opted out")

        manager, _, _ = _manager(runner=runner)
        job, _ = manager.submit(PAYLOAD)
        assert job.status is JobStatus.FAILED
        assert job.error is not None and job.error.startswith("provider_opt_out:")

    def test_value_error_fails_job_as_invalid_request(self) -> None:
        def runner(payload: Any) -> BaseModel:
            raise ValueError("no question provided")

        manager, _, _ = _manager(runner=runner)
        job, _ = manager.submit(PAYLOAD)
        assert job.status is JobStatus.FAILED
        assert job.error is not None and job.error.startswith("invalid_request:")

    def test_unexpected_error_fails_job(self) -> None:
        def runner(payload: Any) -> BaseModel:
            raise RuntimeError("LLM transport exploded")

        manager, _, _ = _manager(runner=runner)
        job, _ = manager.submit(PAYLOAD)
        assert job.status is JobStatus.FAILED
        assert job.error == "forecast_failed: LLM transport exploded"

    def test_lost_claim_skips_execution(self) -> None:
        """If another worker claimed the job first, this one must not run it."""
        calls: list[Any] = []
        executor = DeferredExecutor()

        def runner(payload: Any) -> BaseModel:
            calls.append(payload)
            return _StubResult()

        manager, store, _ = _manager(runner=runner, executor=executor)
        job, _ = manager.submit(PAYLOAD)
        assert store.claim(job.job_id, started_at=T0) is not None  # rival wins
        executor.run_all()
        assert calls == []
        persisted = store.get(job.job_id)
        assert persisted is not None and persisted.status is JobStatus.RUNNING


class TestJobManagerGet:
    def test_get_unknown_returns_none(self) -> None:
        manager, _, _ = _manager()
        assert manager.get("nope") is None

    def test_get_returns_terminal_job_without_waiting(self) -> None:
        manager, _, timeline = _manager()
        job, _ = manager.submit(PAYLOAD)
        fetched = manager.get(job.job_id, wait_s=30.0)
        assert fetched is not None and fetched.status is JobStatus.SUCCEEDED
        assert timeline.sleeps == []

    def test_long_poll_waits_until_deadline_while_queued(self) -> None:
        executor = DeferredExecutor()
        manager, _, timeline = _manager(executor=executor)
        job, _ = manager.submit(PAYLOAD)
        fetched = manager.get(job.job_id, wait_s=2.0)
        assert fetched is not None and fetched.status is JobStatus.QUEUED
        assert timeline.now == T0 + timedelta(seconds=2.0)

    def test_long_poll_returns_early_on_completion(self) -> None:
        executor = DeferredExecutor()
        manager, _, timeline = _manager(executor=executor)
        job, _ = manager.submit(PAYLOAD)

        original_sleep = timeline.sleep

        def sleep_then_finish(seconds: float) -> None:
            original_sleep(seconds)
            executor.run_all()  # the worker completes during the first sleep

        manager._sleep = sleep_then_finish  # noqa: SLF001
        fetched = manager.get(job.job_id, wait_s=60.0)
        assert fetched is not None and fetched.status is JobStatus.SUCCEEDED
        assert len(timeline.sleeps) == 1  # returned on the next check, not at deadline

    def test_wait_is_clamped_to_max(self) -> None:
        executor = DeferredExecutor()
        manager, _, timeline = _manager(executor=executor)
        job, _ = manager.submit(PAYLOAD)
        manager.get(job.job_id, wait_s=10_000.0)
        assert timeline.now - T0 <= timedelta(seconds=MAX_WAIT_S + 1)

    def test_negative_wait_is_treated_as_zero(self) -> None:
        executor = DeferredExecutor()
        manager, _, timeline = _manager(executor=executor)
        job, _ = manager.submit(PAYLOAD)
        fetched = manager.get(job.job_id, wait_s=-5.0)
        assert fetched is not None and fetched.status is JobStatus.QUEUED
        assert timeline.sleeps == []

    def test_nan_wait_is_treated_as_zero(self) -> None:
        """NaN must not reach timedelta (which raises) or poison the clamp."""
        executor = DeferredExecutor()
        manager, _, timeline = _manager(executor=executor)
        job, _ = manager.submit(PAYLOAD)
        fetched = manager.get(job.job_id, wait_s=float("nan"))
        assert fetched is not None and fetched.status is JobStatus.QUEUED
        assert timeline.sleeps == []

    def test_stale_running_job_is_failed_on_poll(self) -> None:
        store = InMemoryJobStore()
        store.create(_job("fj-stale"))
        store.claim("fj-stale", started_at=T0)
        late = Timeline(T0 + timedelta(seconds=3600))
        manager, _, _ = _manager(store=store, timeline=late, stale_after_s=1800.0)
        fetched = manager.get("fj-stale")
        assert fetched is not None
        assert fetched.status is JobStatus.FAILED
        assert fetched.error is not None and "worker lost" in fetched.error
        persisted = store.get("fj-stale")
        assert persisted is not None and persisted.status is JobStatus.FAILED

    def test_fresh_running_job_is_not_failed(self) -> None:
        store = InMemoryJobStore()
        store.create(_job("fj-live"))
        store.claim("fj-live", started_at=T0)
        soon = Timeline(T0 + timedelta(seconds=10))
        manager, _, _ = _manager(store=store, timeline=soon, stale_after_s=1800.0)
        fetched = manager.get("fj-live")
        assert fetched is not None and fetched.status is JobStatus.RUNNING

    def test_queued_orphan_is_revived_by_poll(self) -> None:
        """A queued job from a dead process is re-dispatched (restart safety)."""
        calls: list[Any] = []
        store = InMemoryJobStore()
        store.create(_job("fj-orphan"))

        def runner(payload: Any) -> BaseModel:
            calls.append(payload)
            return _StubResult()

        manager, _, _ = _manager(store=store, runner=runner)
        fetched = manager.get("fj-orphan")
        assert fetched is not None and fetched.status is JobStatus.SUCCEEDED
        assert len(calls) == 1

    def test_inflight_guard_dedupes_dispatch(self) -> None:
        executor = DeferredExecutor()
        manager, _, _ = _manager(executor=executor)
        job, _ = manager.submit(PAYLOAD)
        manager.get(job.job_id)
        manager.get(job.job_id)
        assert len(executor.pending) == 1  # submit dispatched once; polls dedupe


class TestJobManagerThreaded:
    """One integration test of the default ThreadPoolExecutor path (hermetic)."""

    def test_submit_and_poll_with_real_threads(self) -> None:
        release = threading.Event()

        def runner(payload: Any) -> BaseModel:
            assert release.wait(timeout=5.0), "test runner was never released"
            return _StubResult(probability=0.9)

        manager = JobManager(store=InMemoryJobStore(), runner=runner, workers=1)
        try:
            job, created = manager.submit(PAYLOAD)
            assert created is True
            assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            release.set()
            fetched = manager.get(job.job_id, wait_s=5.0)
            assert fetched is not None
            assert fetched.status is JobStatus.SUCCEEDED
            assert fetched.result == {"probability": 0.9}
        finally:
            manager.close()

    def test_close_without_executor_is_noop(self) -> None:
        manager = JobManager(store=InMemoryJobStore(), runner=lambda p: _StubResult())
        manager.close()  # never dispatched: nothing to shut down


def test_missing_timestamps_never_count_as_stale() -> None:
    """A running job with no started_at/created_at cannot be declared stale."""
    store = InMemoryJobStore()
    store.create(ForecastJob(job_id="fj-x", status=JobStatus.QUEUED, request={}, created_at=None))
    store.claim("fj-x", started_at=None)  # type: ignore[arg-type]
    late = Timeline(T0 + timedelta(days=365))
    manager, _, _ = _manager(store=store, timeline=late)
    fetched = manager.get("fj-x")
    assert fetched is not None and fetched.status is JobStatus.RUNNING


def test_submitted_payload_is_copied() -> None:
    """Mutating the caller's payload after submit must not affect the job."""
    manager, store, _ = _manager(executor=DeferredExecutor())
    payload = dict(PAYLOAD)
    job, _ = manager.submit(payload)
    payload["question"] = "mutated"
    persisted = store.get(job.job_id)
    assert persisted is not None
    assert persisted.request["question"] == "Will X ship?"


@pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED])
def test_terminal_statuses_are_terminal(status: JobStatus) -> None:
    from api.jobs import TERMINAL_STATUSES

    assert status in TERMINAL_STATUSES
    assert JobStatus.QUEUED not in TERMINAL_STATUSES
    assert JobStatus.RUNNING not in TERMINAL_STATUSES
