"""Forecast + Deep + intake routes (C10.2).

Two tiers over one surface (CLAUDE.md §9): **DELPHI** (shallow ensemble, the
everyday tier) runs the fixed chain directly; **DELPHI Deep** (maximal
orchestration depth) runs the heuristic conductor. Tier routing is the only
place the *how* differs; both return the identical §9 envelope so a buyer cannot
tell which ran (the routing is proprietary, the evidence is exposed).

The intake surfaces expose the pre-forecast pipeline stages to callers (e.g. a
dashboard routing questions before committing to a forecast): ``classify``
types a question, ``formalize`` returns its normalized resolvable form or a
refusal. Neither records to the registry — the question genesis record is
written by the forecast path only, so the registry never accumulates questions
that were merely previewed.
"""

from __future__ import annotations

from collections.abc import MutableSequence

from api.compliance import filter_providers, should_retain, usage_for
from api.envelope import build_envelope
from api.schema import (
    ClassifyAPIResponse,
    ForecastAPIRequest,
    ForecastAPIResponse,
    FormalizeAPIResponse,
    IntakeAPIRequest,
    build_classify_response,
    build_formalize_response,
    build_response,
)
from conductor.heuristic import HeuristicConductor
from core.registry.store import RegistryStore
from forecaster.chain import Forecaster
from intake.service import IntakeService

__all__ = ["ForecastService"]

# Nominal model-call counts per surface (for usage reporting; not a price).
_TIER_CALLS = {"delphi": 1, "delphi_deep": 2}
_CLASSIFY_CALLS = 1  # one typing call
_FORMALIZE_CALLS = 2  # typing + normalization
_DEFAULT_PROVIDERS: tuple[str, ...] = ("anthropic",)


class ForecastService:
    """Dispatches a request to the right surface and builds the response."""

    def __init__(
        self,
        *,
        forecaster: Forecaster,
        conductor: HeuristicConductor,
        store: RegistryStore,
        intake: IntakeService,
        providers: tuple[str, ...] = _DEFAULT_PROVIDERS,
        price_per_call: float | None = None,
        request_log: MutableSequence[str] | None = None,
    ) -> None:
        self._forecaster = forecaster
        self._conductor = conductor
        self._store = store
        self._intake = intake
        self._providers = providers
        self._price_per_call = price_per_call
        self._request_log = request_log

    def classify(self, request: IntakeAPIRequest) -> ClassifyAPIResponse:
        """Type a question (binary/numeric/multiple-choice/date/unknown)."""
        question, _, _ = self._admit(request)
        classification = self._intake.classify(question)
        usage = usage_for(
            "classify", model_calls=_CLASSIFY_CALLS, price_per_call=self._price_per_call
        )
        return build_classify_response(request, classification, usage)

    def formalize(self, request: IntakeAPIRequest) -> FormalizeAPIResponse:
        """Normalize a question into its resolvable form, or refuse it.

        Runs the full intake gate (classify -> normalize -> refusal) without
        recording anything to the registry.
        """
        question, _, _ = self._admit(request)
        outcome = self._intake.assess(question, as_of=request.as_of_dt())
        usage = usage_for(
            "formalize", model_calls=_FORMALIZE_CALLS, price_per_call=self._price_per_call
        )
        return build_formalize_response(request, outcome, usage)

    def _admit(
        self, request: IntakeAPIRequest | ForecastAPIRequest
    ) -> tuple[str, bool, tuple[str, ...]]:
        """Shared request admission: compliance gate, question, retention logging."""
        options = request.compliance_options()
        permitted = filter_providers(self._providers, options)  # raises if all opted out
        question = request.resolved_question()
        retained = should_retain(options)
        if retained and self._request_log is not None:
            self._request_log.append(question)
        return question, retained, permitted

    def precheck_forecast(self, request: ForecastAPIRequest) -> None:
        """Validate a forecast request without running it (async job admission).

        Raises exactly what the synchronous path would (provider opt-out,
        missing question, malformed as-of) so a doomed job is rejected at
        submit time and never queued to burn model budget. Deliberately free
        of side effects: retention logging happens when :meth:`forecast`
        actually executes the job.
        """
        filter_providers(self._providers, request.compliance_options())
        request.resolved_question()
        request.as_of_dt()

    def forecast(self, request: ForecastAPIRequest) -> ForecastAPIResponse:
        """Route by tier, honor compliance, and return the OpenAI-shaped response."""
        question, retained, permitted = self._admit(request)
        as_of = request.as_of_dt()

        red_team_counter = ""
        workflow = None
        if request.tier == "delphi_deep":
            conducted = self._conductor.conduct(question, as_of=as_of)
            result = conducted.forecast
            red_team_counter = conducted.red_team_counter
            workflow = conducted.workflow.as_dict()
        else:
            result = self._forecaster.forecast(question, as_of=as_of)

        envelope = build_envelope(
            result,
            store=self._store,
            red_team_counter=red_team_counter,
            workflow=workflow,
            retained=retained,
            providers=permitted,
        )
        usage = usage_for(
            request.tier,
            model_calls=_TIER_CALLS[request.tier],
            price_per_call=self._price_per_call,
        )

        finish_reason = "refusal" if envelope.refused else "stop"
        return build_response(request, envelope, usage, finish_reason=finish_reason)
