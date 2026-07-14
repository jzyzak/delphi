"""Forecast + Deep routes (C10.2).

Two tiers over one surface (CLAUDE.md §9): **DELPHI** (shallow ensemble, the
everyday tier) runs the fixed chain directly; **DELPHI Deep** (maximal
orchestration depth) runs the heuristic conductor. Tier routing is the only
place the *how* differs; both return the identical §9 envelope so a buyer cannot
tell which ran (the routing is proprietary, the evidence is exposed).
"""

from __future__ import annotations

from collections.abc import MutableSequence

from api.compliance import filter_providers, should_retain, usage_for
from api.envelope import build_envelope
from api.schema import ForecastAPIRequest, ForecastAPIResponse, build_response
from conductor.heuristic import HeuristicConductor
from core.registry.store import RegistryStore
from forecaster.chain import Forecaster

__all__ = ["ForecastService"]

# Nominal model-call counts per tier (for usage reporting; not a price).
_TIER_CALLS = {"delphi": 1, "delphi_deep": 2}
_DEFAULT_PROVIDERS: tuple[str, ...] = ("anthropic",)


class ForecastService:
    """Dispatches a request to the right tier and builds the §9 response."""

    def __init__(
        self,
        *,
        forecaster: Forecaster,
        conductor: HeuristicConductor,
        store: RegistryStore,
        providers: tuple[str, ...] = _DEFAULT_PROVIDERS,
        price_per_call: float | None = None,
        request_log: MutableSequence[str] | None = None,
    ) -> None:
        self._forecaster = forecaster
        self._conductor = conductor
        self._store = store
        self._providers = providers
        self._price_per_call = price_per_call
        self._request_log = request_log

    def forecast(self, request: ForecastAPIRequest) -> ForecastAPIResponse:
        """Route by tier, honor compliance, and return the OpenAI-shaped response."""
        options = request.compliance_options()
        permitted = filter_providers(self._providers, options)  # raises if all opted out
        question = request.resolved_question()
        as_of = request.as_of_dt()
        retained = should_retain(options)

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
        if retained and self._request_log is not None:
            self._request_log.append(question)

        finish_reason = "refusal" if envelope.refused else "stop"
        return build_response(request, envelope, usage, finish_reason=finish_reason)
