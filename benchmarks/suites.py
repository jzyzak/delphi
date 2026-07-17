"""Retrospective evaluation suite loader (C7 retrospective path).

Turns a benchmark :class:`~benchmarks.base.BenchmarkAdapter` plus a forecasting
function into a fully-assembled :class:`~evaluation.report.EvalContext` that the
``delphi eval`` command renders. The engine here is generic and hermetic (no
network, no LLM): suite-specific fetching and baseline construction live in the
CLI wiring, which injects an adapter, a forecast function, and any extra baselines.

Calibration discipline (CLAUDE.md §2.5): every resolved question is forecast once
to obtain a *raw* (reconciled) probability. The recalibrator + extremization
coefficient are fit ONLY on a calibration split that is disjoint from the scored
(test) split, then applied to the disjoint test split. Calibration questions are
never scored, so the recalibrator is never fit on the data it is judged against.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import structlog

from benchmarks.base import BenchmarkAdapter, scored_records
from core.forecast.calibration import DEFAULT_ALPHA, FrozenCalibration, apply_floor, calibrate
from core.forecast.leakage_judge import LeakageJudge, Trace, TraceComponent
from evaluation.baselines import MARKET_CONSENSUS, Baseline
from evaluation.calibration_split import (
    assign_calibration_split,
    fit_calibration_artifact,
    question_fingerprint,
)
from evaluation.harness import EvalHarness
from evaluation.report import EvalContext, EvalInputs
from evaluation.scoring import ScoredRecord
from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

if TYPE_CHECKING:
    from forecaster.chain import Forecaster

_LOG = structlog.get_logger(__name__)

__all__ = [
    "ForecastFn",
    "QuestionForecast",
    "ResumeStore",
    "build_eval_context",
    "constant_baseline",
    "filter_records_by_source",
    "forecaster_fn",
    "records_baseline",
    "run_key_for",
    "sample_records",
]


@dataclass(frozen=True)
class QuestionForecast:
    """One question's raw (pre-recalibration) forecast plus its audit traces."""

    accepted: bool
    raw_probability: float | None
    traces: tuple[Trace, ...] = ()


class ForecastFn(Protocol):
    """Forecast a question as of a knowledge ceiling.

    ``metadata`` (optional) threads benchmark identity + freeze value onto the
    recorded question so the registry record is later resolvable
    (`delphi resolve --suite ...`). Kept as a plain-callable protocol so tests
    can supply a deterministic stub with no LLM/network.
    """

    def __call__(
        self,
        text: str,
        as_of: datetime,
        metadata: Mapping[str, Any] | None = None,
        /,
    ) -> QuestionForecast: ...


def _traces_from_result(
    evidence: Sequence[object], rationale: str, *, forecast_id: str, as_of: datetime
) -> tuple[Trace, ...]:
    """Build leakage-audit traces from a forecast's evidence and rationale."""
    traces: list[Trace] = []
    for item in evidence:
        snippet = getattr(item, "snippet", None)
        knowledge_time = getattr(item, "knowledge_time", None)
        if snippet is None or knowledge_time is None:
            continue
        traces.append(
            Trace(
                component=TraceComponent.SEARCH,
                as_of=knowledge_time,
                text=str(snippet),
                forecast_id=forecast_id,
                metadata={"source": str(getattr(item, "source", ""))},
            )
        )
    if rationale:
        traces.append(
            Trace(
                component=TraceComponent.SUPERVISOR,
                as_of=as_of,
                text=rationale,
                forecast_id=forecast_id,
            )
        )
    return tuple(traces)


def records_baseline(
    records: Sequence[Mapping[str, object]],
    *,
    source: str,
    value_key: str = "freeze_value",
    name: str = MARKET_CONSENSUS,
) -> Baseline:
    """Build a baseline from a per-question value carried in fetched records.

    Used for benchmarks whose crowd/market value is dropped by the adapter (e.g.
    ForecastBench's ``freeze_datetime_value``): the value is read straight off the
    fetched record and keyed to the benchmark ``question_id`` (``source:id``).
    """
    predictions: dict[str, float] = {}
    for record in records:
        value = record.get(value_key)
        raw_id = record.get("id")
        if value is None or raw_id is None:
            continue
        try:
            predictions[f"{source}:{raw_id}"] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return Baseline(name=name, predictions=predictions)


def constant_baseline(
    records: Sequence[Mapping[str, object]],
    *,
    source: str,
    value: float = 0.25,
    name: str = "uninformed_0.25",
) -> Baseline:
    """A constant-probability baseline over every fetched question.

    ``value=0.25`` is the uninformed reference for binary Brier: a forecaster
    with zero information scores 0.25 by always saying 0.5. Beating it says
    the system knows *something*; a no-search arm beating it by a large margin
    on post-cutoff questions is the parametric-leakage smell.
    """
    predictions = {
        f"{source}:{record['id']}": value for record in records if record.get("id") is not None
    }
    return Baseline(name=name, predictions=predictions)


def filter_records_by_source(
    records: Sequence[dict[str, Any]], sources: Sequence[str]
) -> list[dict[str, Any]]:
    """Keep only records whose composite id prefix is in ``sources``.

    ForecastBench ids are ``<source>-<hash-or-ref>``; the prefix identifies the
    question family (fred, dbnomics, acled, ...). Used for targeted domain
    evals — the run still goes through the guarded harness like any other.
    """
    wanted = {source.strip().lower() for source in sources if source.strip()}
    if not wanted:
        msg = "at least one source is required to filter by."
        raise ValueError(msg)
    return [
        record for record in records if str(record.get("id", "")).split("-", 1)[0].lower() in wanted
    ]


def sample_records(
    records: Sequence[dict[str, Any]], *, n: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Deterministic stratified subsample of fetched benchmark records.

    Strata are question sources (the id prefix before the first ``-``, per the
    ForecastBench composite id convention); allocation is proportional with
    largest-remainder rounding, selection within a stratum is a seeded draw
    over id-sorted records. Replaces hand-truncated question files: the same
    ``(n, seed)`` always yields the same subsample, so ablation arms run on
    identical questions.
    """
    if n < 1:
        msg = f"n must be >= 1, got {n!r}"
        raise ValueError(msg)
    if n >= len(records):
        return list(records)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(records, key=lambda r: str(r.get("id", ""))):
        source = str(record.get("id", "")).split("-", 1)[0]
        by_source.setdefault(source, []).append(record)

    # Proportional allocation with largest-remainder rounding. Since n < total,
    # int(exact) < len(group) for every stratum, so a +1 remainder grant can
    # never overflow a stratum and the shortfall (= sum of fractional parts)
    # is always absorbed within this single pass.
    total = len(records)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for source, group in sorted(by_source.items()):
        exact = n * len(group) / total
        quotas[source] = int(exact)
        remainders.append((exact - int(exact), source))
    shortfall = n - sum(quotas.values())
    # Largest remainder first; ties break by source name (deterministic).
    for _, source in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if shortfall <= 0:
            break
        quotas[source] += 1
        shortfall -= 1

    rng = np.random.default_rng(seed)
    sampled: list[dict[str, Any]] = []
    for source, group in sorted(by_source.items()):
        take = quotas[source]
        if take == 0:
            continue
        indices = sorted(rng.choice(len(group), size=take, replace=False).tolist())
        sampled.extend(group[i] for i in indices)
    return sampled


class ResumeStore:
    """Durable per-question forecast log so an interrupted eval run resumes.

    JSONL, append-only, flushed per entry: a header line pins the run identity
    (question-set fingerprint + a caller-supplied tag), then one line per
    forecast (accepted or refused). On restart with the same file, already
    forecast questions are replayed instead of re-paid; a header mismatch
    fails hard — resuming a *different* run's file would silently mix
    incomparable forecasts. Failed questions are never persisted, so they are
    retried on resume.
    """

    def __init__(self, path: str | Path, *, run_key: str, tag: str = "") -> None:
        self._path = Path(path).expanduser()
        self._run_key = run_key
        self._tag = tag
        self._entries: dict[str, QuestionForecast] = {}
        if self._path.exists():
            self._load()
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._append({"kind": "header", "run_key": run_key, "tag": tag})

    @property
    def entries(self) -> Mapping[str, QuestionForecast]:
        return dict(self._entries)

    def _load(self) -> None:
        lines = self._path.read_text(encoding="utf-8").splitlines()
        if not lines:
            self._append({"kind": "header", "run_key": self._run_key, "tag": self._tag})
            return
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            msg = f"resume file {self._path} has a corrupt header line."
            raise ValueError(msg) from exc
        if header.get("kind") != "header" or header.get("run_key") != self._run_key:
            msg = (
                f"resume file {self._path} belongs to a different run "
                f"(tag={header.get('tag')!r}); refusing to mix forecasts across "
                "runs. Use a fresh --resume-file per run configuration."
            )
            raise ValueError(msg)
        torn_tail = False
        for i, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from an interruption mid-write is expected;
                # anything else is corruption. The torn question is simply
                # re-forecast.
                if i == len(lines):
                    _LOG.warning("suites.resume_torn_tail_dropped", path=str(self._path))
                    torn_tail = True
                    continue
                msg = f"resume file {self._path} is corrupt at line {i}."
                raise ValueError(msg) from None
            traces = tuple(Trace.model_validate(t) for t in entry.get("traces", ()))
            self._entries[str(entry["question_id"])] = QuestionForecast(
                accepted=bool(entry["accepted"]),
                raw_probability=entry.get("raw_probability"),
                traces=traces,
            )
        if torn_tail:
            # Rewrite the file without the torn fragment so later appends
            # start on a fresh line instead of concatenating onto garbage.
            self._rewrite()

    def _rewrite(self) -> None:
        with self._path.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"kind": "header", "run_key": self._run_key, "tag": self._tag},
                    sort_keys=True,
                )
                + "\n"
            )
            for question_id, forecast in self._entries.items():
                fh.write(
                    json.dumps(
                        {
                            "kind": "forecast",
                            "question_id": question_id,
                            "accepted": forecast.accepted,
                            "raw_probability": forecast.raw_probability,
                            "traces": [t.model_dump(mode="json") for t in forecast.traces],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            fh.flush()

    def _append(self, payload: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
            fh.flush()

    def record(self, question_id: str, forecast: QuestionForecast) -> None:
        self._entries[question_id] = forecast
        self._append(
            {
                "kind": "forecast",
                "question_id": question_id,
                "accepted": forecast.accepted,
                "raw_probability": forecast.raw_probability,
                "traces": [t.model_dump(mode="json") for t in forecast.traces],
            }
        )


def run_key_for(resolved_ids: Sequence[str], *, tag: str = "") -> str:
    """Stable identity for a run: the exact scored question set + arm tag."""
    digest = hashlib.sha256()
    for qid in sorted(resolved_ids):
        digest.update(qid.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(tag.encode("utf-8"))
    return digest.hexdigest()


def forecaster_fn(forecaster: Forecaster) -> ForecastFn:
    """Adapt a real :class:`Forecaster` into a :data:`ForecastFn`.

    Uses the *raw reconciled* probability (before recalibration/extremization) so
    the suite loader can fit and apply calibration itself on disjoint splits — the
    forecaster's own recalibrator is intentionally not double-applied here.
    """

    def _fn(
        text: str, as_of: datetime, metadata: Mapping[str, Any] | None = None
    ) -> QuestionForecast:
        result = forecaster.forecast(text, as_of=as_of, metadata=metadata)
        if not result.accepted or result.calibrated is None:
            return QuestionForecast(accepted=False, raw_probability=None)
        traces = _traces_from_result(
            result.evidence,
            result.rationale,
            forecast_id=result.question_id or "",
            as_of=as_of,
        )
        return QuestionForecast(
            accepted=True,
            raw_probability=result.calibrated.raw_probability,
            traces=traces,
        )

    return _fn


def _fit_overlap_ids(
    calibration: FrozenCalibration, question_ids: Sequence[str], questions: Mapping[str, Any]
) -> set[str]:
    """Question ids that fingerprint into the artifact's fit set (§2.5)."""
    fingerprints = set(calibration.fitted_meta.get("question_fingerprints", ()))
    if not fingerprints:
        return set()
    overlap: set[str] = set()
    for qid in question_ids:
        candidates = [question_fingerprint(qid)]
        question = questions.get(qid)
        if question is not None:
            candidates.append(question_fingerprint(question.text))
        if any(fp in fingerprints for fp in candidates):
            overlap.add(qid)
    return overlap


def _assert_disjoint_from_fit(
    calibration: FrozenCalibration, question_ids: Sequence[str], questions: Mapping[str, Any]
) -> None:
    """Raise if any scored question fingerprints into the artifact's fit set (§2.5)."""
    overlap = _fit_overlap_ids(calibration, question_ids, questions)
    if overlap:
        qid = sorted(overlap)[0]
        msg = (
            f"question {qid!r} appears in the calibration artifact's fit set: "
            "scoring it would leak the calibration corpus into the scored set "
            "(§2.5). Use a disjoint question set or refit the artifact."
        )
        raise ValueError(msg)


def build_eval_context(
    adapter: BenchmarkAdapter,
    forecast_fn: ForecastFn,
    *,
    harness: EvalHarness,
    judge: LeakageJudge | None = None,
    calibration_fraction: float = 0.5,
    holdout_ids: Iterable[str] = (),
    seed: int = 0,
    extra_baselines: Sequence[Baseline] = (),
    calibration: FrozenCalibration | None = None,
    resume_path: str | Path | None = None,
    resume_tag: str = "",
    exclude_fit_questions: bool = False,
) -> EvalContext:
    """Assemble an :class:`EvalContext` for one benchmark suite.

    Default mode: resolved binary questions are forecast once, split into
    disjoint calibration/test sets, calibrated using only the calibration
    split, scored on the test split. Artifact mode (``calibration`` provided):
    the pre-fitted map — fit on a *separate historical corpus* — is applied to
    ALL accepted questions and all of them are scored; a fingerprint check
    raises if any scored question was in the artifact's fit set (§2.5), unless
    ``exclude_fit_questions`` is set, in which case fit-set questions are
    dropped BEFORE forecasting (never scored, never charged) and the exclusion
    count is reported in the calibration provenance.
    """
    questions = {q.question_id: q for q in adapter.questions()}
    outcome_of = {r.question_id: r.resolved_value for r in adapter.resolutions()}
    # The scheduled resolution date is part of the question's own definition
    # (many benchmark texts literally reference {resolution_date}); it is the
    # as-of-safe horizon the series estimator needs. close_time when the
    # adapter carries one, else the benchmark resolution's date.
    resolution_date_of = {r.question_id: r.resolved_at for r in adapter.resolutions()}
    resolved_ids = [
        r.question_id
        for r in adapter.resolutions()
        if r.resolved_value in (0.0, 1.0) and r.question_id in questions
    ]

    n_fit_excluded = 0
    if calibration is not None and exclude_fit_questions:
        overlap = _fit_overlap_ids(calibration, resolved_ids, questions)
        if overlap:
            n_fit_excluded = len(overlap)
            _LOG.warning(
                "suites.fit_overlap_excluded",
                n_excluded=n_fit_excluded,
                n_resolved=len(resolved_ids),
            )
            resolved_ids = [qid for qid in resolved_ids if qid not in overlap]
            if not resolved_ids:
                msg = (
                    "every resolved question is in the calibration artifact's "
                    "fit set (§2.5); use a disjoint question set."
                )
                raise ValueError(msg)

    resume: ResumeStore | None = None
    if resume_path is not None:
        resume = ResumeStore(
            resume_path, run_key=run_key_for(resolved_ids, tag=resume_tag), tag=resume_tag
        )
        if resume.entries:
            _LOG.info(
                "suites.resuming_from_store",
                n_resumed=len(resume.entries),
                n_resolved=len(resolved_ids),
            )

    raw: dict[str, float] = {}
    traces_by_id: dict[str, tuple[Trace, ...]] = {}
    failed: list[str] = []
    for qid in resolved_ids:
        question = questions[qid]
        forecast = resume.entries.get(qid) if resume is not None else None
        if forecast is None:
            # Benchmark identity rides the recorded question's metadata so the
            # registry record is later resolvable against the benchmark's
            # ground truth (mirrors HarvestJob) — an unresolvable forecast can
            # never become calibration corpus.
            metadata: dict[str, Any] = {
                BENCHMARK_QUESTION_ID_KEY: qid,
                "benchmark_source": question.source,
                "benchmark_external_id": question.external_id,
            }
            freeze = question.metadata.get(
                "freeze_value", question.metadata.get("community_prediction")
            )
            if freeze is not None:
                # The market/crowd value at the freeze (== the as-of) — the
                # forecaster injects it as as-of evidence (AIA-style anchor).
                metadata["market_freeze_value"] = float(freeze)
            resolution_date = question.close_time or resolution_date_of.get(qid)
            if resolution_date is not None:
                # Gives the series-threshold estimator its horizon.
                metadata["resolution_date"] = resolution_date.isoformat()
            try:
                forecast = forecast_fn(question.text, question.as_of, metadata)
            except Exception:  # noqa: BLE001 - one question must not kill the run
                # A transient failure (network drop, provider outage) on one
                # question must not destroy an hours-long run: log it, skip it,
                # and report the drop count loudly — never silently. Failures
                # are NOT persisted, so a resumed run retries them.
                _LOG.exception("suites.question_forecast_failed", question_id=qid)
                failed.append(qid)
                continue
            if resume is not None:
                resume.record(qid, forecast)
        if not forecast.accepted or forecast.raw_probability is None:
            continue  # refused questions do not enter the scored set.
        raw[qid] = forecast.raw_probability
        traces_by_id[qid] = forecast.traces
    if failed:
        _LOG.warning(
            "suites.questions_dropped_after_forecast_failure",
            n_failed=len(failed),
            n_resolved=len(resolved_ids),
            failed_question_ids=failed,
        )

    if not raw:
        msg = "no accepted forecasts to evaluate for this suite."
        raise ValueError(msg)

    accepted_ids = sorted(raw)
    calibration_provenance: dict[str, Any] | None = None
    if calibration is not None:
        # Artifact mode: pre-fitted on a disjoint historical corpus — every
        # accepted question is scored.
        _assert_disjoint_from_fit(calibration, accepted_ids, questions)
        test_ids = accepted_ids
        forecasts = {
            qid: apply_floor(
                calibrate(calibration.apply(raw[qid]), alpha=calibration.alpha),
                calibration.floor,
            )
            for qid in test_ids
        }
        calibration_provenance = dict(calibration.provenance)
        if n_fit_excluded:
            calibration_provenance["excluded_fit_overlap"] = n_fit_excluded
    else:
        calibration_ids = assign_calibration_split(
            accepted_ids, holdout_ids=holdout_ids, fraction=calibration_fraction, seed=seed
        )
        test_ids = [qid for qid in accepted_ids if qid not in calibration_ids]
        if not test_ids:
            msg = "calibration split consumed every question; lower calibration_fraction."
            raise ValueError(msg)

        if calibration_ids:
            cal_records = [
                ScoredRecord(
                    question_id=qid,
                    domain=questions[qid].domain,
                    probability=raw[qid],
                    outcome=outcome_of[qid],
                )
                for qid in sorted(calibration_ids)
            ]
            artifact = fit_calibration_artifact(cal_records)
            apply_calibration = artifact.apply
            calibration_provenance = {
                "recalibrator": artifact.recalibrator.method,
                "alpha": artifact.alpha,
                "floor": artifact.floor,
                "fallback": artifact.fallback,
                "n": artifact.recalibrator.n,
                "fitted": True,
                "source": "within-run split",
            }
        else:

            def apply_calibration(probability: float) -> float:
                return calibrate(probability, alpha=DEFAULT_ALPHA)

        forecasts = {qid: apply_calibration(raw[qid]) for qid in test_ids}

    records = scored_records(forecasts, adapter)
    traces = tuple(trace for qid in test_ids for trace in traces_by_id.get(qid, ()))
    inputs = EvalInputs(
        records=records,
        baselines=tuple(extra_baselines),
        traces=traces,
        calibration_provenance=calibration_provenance,
    )
    return EvalContext(inputs=inputs, harness=harness, judge=judge)
