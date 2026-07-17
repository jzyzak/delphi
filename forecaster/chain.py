"""End-to-end forecast chain (C4.7).

Composes the §3 pipeline into a single ``Forecaster.forecast(question, as_of)``:
intake -> as-of search -> base rate -> decomposition -> inside-view ensemble ->
supervisor reconcile -> calibrate + uncertainty -> leakage gate -> registry
write. There is no silent forecast path: an accepted question always writes a
complete ``(question, evidence_set, forecast)`` record, and ``as_of`` is an
explicit input threaded through every stage (no ``now()``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.forecast.bayesian import EvidenceLikelihoodLLM
from core.forecast.calibration import CalibratedForecast, FrozenCalibration
from core.forecast.leakage_judge import LeakageJudge
from core.forecast.llm import ForecastLLM
from core.forecast.search import AsOfSearcher, Evidence
from core.forecast.supervisor import SupervisorLLM
from core.forecast.uncertainty import Uncertainty
from core.pit.models import ensure_utc
from core.registry.fingerprint import content_hash
from core.registry.store import RegistryStore
from forecaster.record import (
    build_evidence_set_input,
    build_forecast_input,
    build_rationale,
)
from forecaster.stages.aggregate import SupervisorTuning, reconcile
from forecaster.stages.base_rate import estimate_base_rate
from forecaster.stages.calibrate import Recalibrator, calibrate_reconciled
from forecaster.stages.decompose import decompose_question
from forecaster.stages.inside_view import (
    assemble_bayesian_ensemble,
    assemble_ensemble,
    build_forecast_content,
    build_subset_draw_requests,
)
from forecaster.stages.leakage_gate import LeakageGateResult, run_leakage_gate
from forecaster.stages.series_estimate import SeriesEvidenceEstimator
from intake.llm import StructuredLLM
from intake.refusal import RefusalDecision
from intake.service import IntakeService

__all__ = ["MARKET_FREEZE_SOURCE", "ForecastResult", "Forecaster", "build_evidence_query"]


MARKET_FREEZE_SOURCE = "market_freeze"


def _market_freeze_evidence(
    metadata: Mapping[str, Any] | None, as_of: datetime
) -> tuple[Evidence, ...]:
    """Synthetic evidence carrying the market/crowd freeze value, if provided.

    As-of-safe by construction: the freeze value is defined AT the forecast
    time itself (``knowledge_time == as_of``). Search quality dominates model
    choice, and the market's own estimate is the single highest-value evidence
    item for market-priced questions (AIA credits it at roughly half of search).
    """
    if not metadata:
        return ()
    raw = metadata.get("market_freeze_value")
    if raw is None:
        return ()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return ()
    snippet = (
        f"Market/crowd freeze value for this exact question, recorded at the "
        f"forecast time ({as_of.date().isoformat()}): {value:g}. For "
        f"market-priced questions this is the crowd's probability estimate; "
        f"for data-series questions it is the latest observed value of the "
        f"underlying series at the freeze."
    )
    return (
        Evidence(
            snippet=snippet,
            source=MARKET_FREEZE_SOURCE,
            source_id=MARKET_FREEZE_SOURCE,
            knowledge_time=as_of,
            score=1.0,
        ),
    )


def build_evidence_query(text: str, entities: Sequence[str], horizon: str | None) -> str:
    """Compact evidence-search query: named entities + horizon when available.

    Full question sentences retrieve poorly from keyword-oriented providers
    (Wikipedia search, GDELT); the entities extracted at intake retrieve well
    across every provider. Falls back to the canonical text when intake found
    no entities.
    """
    names = [e.strip() for e in entities if e.strip()]
    if not names:
        return text
    parts = [*names]
    if horizon and horizon.strip():
        parts.append(horizon.strip())
    return " ".join(parts)


@dataclass(frozen=True)
class ForecastResult:
    """The outcome of a forecast request — accepted (with a record) or refused."""

    accepted: bool
    question_id: str | None
    forecast_id: str | None
    probability: float | None
    calibrated: CalibratedForecast | None
    uncertainty: Uncertainty | None
    evidence: tuple[Evidence, ...]
    leakage: LeakageGateResult | None
    quarantined: bool
    rationale: str
    refusal: RefusalDecision | None


class Forecaster:
    """The fixed-pipeline forecaster (CLAUDE.md §3)."""

    def __init__(
        self,
        *,
        intake: IntakeService,
        searcher: AsOfSearcher,
        reasoning_llm: StructuredLLM,
        forecast_llm: ForecastLLM,
        supervisor_llm: SupervisorLLM,
        leakage_judge: LeakageJudge,
        registry_store: RegistryStore,
        recalibrator: Recalibrator | None = None,
        calibration: FrozenCalibration | None = None,
        aggregator: str = "log_odds_trimmed_mean",
        runs_per_agent: int = 1,
        evidence_likelihood_llm: EvidenceLikelihoodLLM | None = None,
        bayesian_draws: int = 10,
        supervisor_tuning: SupervisorTuning | None = None,
        evidence_subset_fraction: float = 1.0,
        max_subquestion_searches: int = 0,
        series_estimator: SeriesEvidenceEstimator | None = None,
    ) -> None:
        self._intake = intake
        self._searcher = searcher
        self._reasoning_llm = reasoning_llm
        self._forecast_llm = forecast_llm
        self._supervisor_llm = supervisor_llm
        self._judge = leakage_judge
        self._store = registry_store
        # A fitted artifact bundles its own recalibrator + alpha + floor and
        # takes precedence over a bare recalibrator (which keeps DEFAULT_ALPHA).
        self._recalibrator = calibration if calibration is not None else recalibrator
        self._calibration = calibration
        self._aggregator = aggregator
        self._runs_per_agent = runs_per_agent
        self._likelihood_llm = evidence_likelihood_llm
        self._bayesian_draws = bayesian_draws
        self._supervisor_tuning = supervisor_tuning
        self._evidence_subset_fraction = evidence_subset_fraction
        if max_subquestion_searches < 0:
            msg = f"max_subquestion_searches must be >= 0, got {max_subquestion_searches!r}"
            raise ValueError(msg)
        self._max_subquestion_searches = max_subquestion_searches
        self._series_estimator = series_estimator

    def _model_provenance(self) -> dict[str, object]:
        return {
            "forecast_llm": {
                "model_version": self._forecast_llm.model_version,
                "prompt_version": self._forecast_llm.prompt_version,
            },
            "supervisor_llm": {
                "model_version": self._supervisor_llm.model_version,
                "prompt_version": self._supervisor_llm.prompt_version,
            },
            "leakage_judge": {
                "model_version": self._judge.model_version,
                "prompt_version": self._judge.prompt_version,
            },
        }

    def forecast(
        self,
        question_text: str,
        *,
        as_of: datetime,
        metadata: Mapping[str, Any] | None = None,
    ) -> ForecastResult:
        """Form a calibrated forecast as of ``as_of`` and write a complete record.

        ``metadata`` is threaded to intake and recorded on the question (e.g. a
        benchmark question id so the live loop can resolve it later).
        """
        ceiling = ensure_utc(as_of)
        outcome = self._intake.intake(question_text, as_of=ceiling, metadata=metadata)
        if not outcome.accepted or outcome.resolvable is None or outcome.question_id is None:
            return ForecastResult(
                accepted=False,
                question_id=None,
                forecast_id=None,
                probability=None,
                calibrated=None,
                uncertainty=None,
                evidence=(),
                leakage=None,
                quarantined=False,
                rationale="",
                refusal=outcome.refusal,
            )

        question_id = outcome.question_id
        query = outcome.resolvable.text
        # Retrieval uses a compact entity+horizon query (keyword providers
        # retrieve poorly on full sentences); reasoning keeps the full text.
        search_query = build_evidence_query(
            query, outcome.resolvable.entities, outcome.classification.horizon
        )
        evidence = self._gather_evidence(search_query, ceiling)
        anchors = _market_freeze_evidence(metadata, ceiling)
        if self._series_estimator is not None:
            # Deterministic quantitative reference class for series-direction
            # questions — computed from as-of history, arithmetic not judgment.
            anchors = (*anchors, *self._series_estimator.evidence(metadata, as_of=ceiling))
        if anchors:
            # Injected FIRST so every downstream stage (base rate, ensemble
            # draws, supervisor) sees the anchors alongside retrieved evidence.
            evidence = (*anchors, *evidence)

        base_rate = estimate_base_rate(query, evidence, llm=self._reasoning_llm, as_of=ceiling)
        decomposition = decompose_question(query, llm=self._reasoning_llm)
        # Decomposition-seeded retrieval: each sub-question is its own as-of
        # search, merged (deduplicated) into the evidence the ensemble reads.
        # The base rate above intentionally anchors on the primary retrieval.
        if self._max_subquestion_searches > 0 and decomposition.sub_questions:
            evidence = self._merge_evidence(
                evidence,
                [
                    self._gather_evidence(sub.text, ceiling)
                    for sub in decomposition.sub_questions[: self._max_subquestion_searches]
                ],
            )
        content = build_forecast_content(query, base_rate, decomposition, evidence)
        if self._likelihood_llm is not None:
            # Bayesian path (§3): prior = reference-class base rate; evidence
            # log-LRs elicited per draw and combined in log-odds space.
            ensemble = assemble_bayesian_ensemble(
                self._likelihood_llm,
                content=content,
                base_rate=base_rate,
                knowledge_time=ceiling,
                n=self._bayesian_draws,
                aggregator=self._aggregator,  # type: ignore[arg-type]
            )
        else:
            requests = build_subset_draw_requests(
                question=query,
                base_rate=base_rate,
                decomposition=decomposition,
                evidence=evidence,
                runs_per_agent=self._runs_per_agent,
                subset_fraction=self._evidence_subset_fraction,
            )
            ensemble = assemble_ensemble(
                self._forecast_llm,
                requests,
                aggregator=self._aggregator,  # type: ignore[arg-type]
                knowledge_time=ceiling,
            )
        reconciled = reconcile(
            ensemble,
            searcher=self._searcher,
            supervisor_llm=self._supervisor_llm,
            tuning=self._supervisor_tuning,
        )
        if self._calibration is not None:
            calibrated, uncertainty = calibrate_reconciled(
                reconciled,
                recalibrator=self._recalibrator,
                alpha=self._calibration.alpha,
                floor=self._calibration.floor,
            )
        else:
            calibrated, uncertainty = calibrate_reconciled(
                reconciled, recalibrator=self._recalibrator
            )
        leakage = run_leakage_gate(
            ensemble,
            reconciled,
            judge=self._judge,
            forecast_id=question_id,
            evidence=evidence,
        )

        evidence_set_id = self._store.record_evidence_set(
            build_evidence_set_input(question_id, ceiling, evidence)
        )
        rationale = build_rationale(base_rate, decomposition, reconciled, calibrated)
        forecast_input = build_forecast_input(
            question_id=question_id,
            as_of=ceiling,
            evidence_set_id=evidence_set_id,
            calibrated=calibrated,
            uncertainty=uncertainty,
            ensemble=ensemble,
            reconciled=reconciled,
            base_rate=base_rate,
            decomposition=decomposition,
            leakage=leakage,
            rationale=rationale,
            model_provenance=self._model_provenance(),
            repro_handle={
                "as_of": ceiling.isoformat(),
                "aggregator": self._aggregator,
                "runs_per_agent": self._runs_per_agent,
                "evidence_subset_fraction": self._evidence_subset_fraction,
                "max_subquestion_searches": self._max_subquestion_searches,
                "search_config": getattr(self._searcher, "search_config", None),
                "content_hash": content_hash(content),
                "n_evidence": len(evidence),
                "calibration_artifact_hash": (
                    self._calibration.artifact_hash if self._calibration is not None else None
                ),
            },
        )
        forecast_id = self._store.record_forecast(forecast_input)

        return ForecastResult(
            accepted=True,
            question_id=question_id,
            forecast_id=forecast_id,
            probability=calibrated.calibrated_probability,
            calibrated=calibrated,
            uncertainty=uncertainty,
            evidence=evidence,
            leakage=leakage,
            quarantined=leakage.flagged,
            rationale=rationale,
            refusal=None,
        )

    def _gather_evidence(self, query: str, as_of: datetime) -> tuple[Evidence, ...]:
        evidence: Sequence[Evidence] = self._searcher.as_of_search(query, as_of=as_of)
        for item in evidence:
            if item.knowledge_time > as_of:  # pragma: no cover - searcher guarantees this
                msg = "as_of_search returned evidence dated after the as-of ceiling."
                raise RuntimeError(msg)
        return tuple(evidence)

    @staticmethod
    def _merge_evidence(
        primary: Sequence[Evidence], extra_batches: Sequence[Sequence[Evidence]]
    ) -> tuple[Evidence, ...]:
        """Merge retrieval passes, deduplicating by (source, source_id, snippet)."""
        merged: dict[tuple[str, str, str], Evidence] = {
            (item.source, item.source_id, item.snippet): item for item in primary
        }
        for batch in extra_batches:
            for item in batch:
                merged.setdefault((item.source, item.source_id, item.snippet), item)
        return tuple(merged.values())
