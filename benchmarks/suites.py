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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from benchmarks.base import BenchmarkAdapter, scored_records
from common.llm.errors import LLMError
from core.forecast.calibration import DEFAULT_ALPHA, calibrate
from core.forecast.leakage_judge import LeakageJudge, Trace, TraceComponent
from evaluation.baselines import MARKET_CONSENSUS, Baseline
from evaluation.calibration_split import assign_calibration_split, fit_calibration_artifact
from evaluation.harness import EvalHarness
from evaluation.report import EvalContext, EvalInputs
from evaluation.scoring import ScoredRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from forecaster.chain import Forecaster

__all__ = [
    "ForecastFn",
    "QuestionForecast",
    "build_eval_context",
    "forecaster_fn",
    "records_baseline",
]


@dataclass(frozen=True)
class QuestionForecast:
    """One question's raw (pre-recalibration) forecast plus its audit traces."""

    accepted: bool
    raw_probability: float | None
    traces: tuple[Trace, ...] = ()


# A function that forecasts a question as of a knowledge ceiling. Kept as a plain
# callable so tests can supply a deterministic stub with no LLM/network.
ForecastFn = Callable[[str, datetime], QuestionForecast]


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


def forecaster_fn(forecaster: Forecaster) -> ForecastFn:
    """Adapt a real :class:`Forecaster` into a :data:`ForecastFn`.

    Uses the *raw reconciled* probability (before recalibration/extremization) so
    the suite loader can fit and apply calibration itself on disjoint splits — the
    forecaster's own recalibrator is intentionally not double-applied here.
    """

    def _fn(text: str, as_of: datetime) -> QuestionForecast:
        result = forecaster.forecast(text, as_of=as_of)
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
) -> EvalContext:
    """Assemble an :class:`EvalContext` for one benchmark suite.

    Resolved binary questions are forecast once, split into disjoint
    calibration/test sets, calibrated using only the calibration split, scored on
    the test split, and paired with the supplied baselines and leakage traces.
    """
    questions = {q.question_id: q for q in adapter.questions()}
    outcome_of = {r.question_id: r.resolved_value for r in adapter.resolutions()}
    resolved_ids = [
        r.question_id
        for r in adapter.resolutions()
        if r.resolved_value in (0.0, 1.0) and r.question_id in questions
    ]

    logger = structlog.get_logger(__name__)
    raw: dict[str, float] = {}
    traces_by_id: dict[str, tuple[Trace, ...]] = {}
    errored: list[str] = []
    for qid in resolved_ids:
        question = questions[qid]
        try:
            forecast = forecast_fn(question.text, question.as_of)
        except LLMError as exc:
            # A single bad question (safety refusal, provider failure after
            # retries) must not kill a multi-hour suite run. Treat it like a
            # refusal: log, exclude from the scored set, keep going.
            logger.warning("eval_question_skipped", question_id=qid, error=str(exc))
            errored.append(qid)
            continue
        if not forecast.accepted or forecast.raw_probability is None:
            continue  # refused questions do not enter the scored set.
        raw[qid] = forecast.raw_probability
        traces_by_id[qid] = forecast.traces
    if errored:
        logger.warning("eval_questions_skipped_total", count=len(errored))

    if not raw:
        msg = "no accepted forecasts to evaluate for this suite."
        raise ValueError(msg)

    accepted_ids = sorted(raw)
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
        recalibrate = artifact.recalibrator.apply
        alpha = artifact.alpha
    else:
        recalibrate = None
        alpha = DEFAULT_ALPHA

    forecasts: dict[str, float] = {}
    for qid in test_ids:
        raw_p = raw[qid]
        recalibrated = recalibrate(raw_p) if recalibrate is not None else raw_p
        forecasts[qid] = calibrate(recalibrated, alpha=alpha)

    records = scored_records(forecasts, adapter)
    traces = tuple(trace for qid in test_ids for trace in traces_by_id.get(qid, ()))
    inputs = EvalInputs(
        records=records,
        baselines=tuple(extra_baselines),
        traces=traces,
    )
    return EvalContext(inputs=inputs, harness=harness, judge=judge)
