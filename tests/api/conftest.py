"""Shared fixtures for the API tests (hermetic: fixture LLMs + fixture search)."""

from __future__ import annotations

from collections.abc import Callable, MutableSequence
from datetime import UTC, datetime

import pytest

from api.routes import ForecastService
from api.server import DelphiApp
from conductor.heuristic import HeuristicConductor
from core.forecast.leakage_judge import FixtureLeakageJudgeLLM, LeakageJudge
from core.forecast.llm import FixtureForecastLLM
from core.forecast.search import Evidence, FixtureAsOfSearch
from core.forecast.supervisor import FixtureSupervisorLLM
from core.registry.store import InMemoryRegistryStore
from forecaster.chain import Forecaster
from intake.llm import FixtureStructuredLLM
from intake.service import IntakeService

AS_OF = "2024-06-01T00:00:00+00:00"

_CLASSIFY = {"question_type": "binary", "entities": ["X"]}
_NORMALIZE = {
    "canonical_text": "Will X ship by 2025?",
    "domain": "tech",
    "resolution_criteria": "Resolves YES on GA announcement.",
    "close_time": "2025-06-01T00:00:00+00:00",
    "resolvable": True,
}


def _forecaster(store: InMemoryRegistryStore, *, classify: dict = _CLASSIFY) -> Forecaster:
    return Forecaster(
        intake=IntakeService(llm=FixtureStructuredLLM([classify, _NORMALIZE]), store=store),
        searcher=FixtureAsOfSearch(
            default=(
                Evidence(
                    snippet="a strong prior signal",
                    source="hosted",
                    source_id="http://example/a",
                    knowledge_time=datetime(2024, 1, 1, tzinfo=UTC),
                    score=0.5,
                ),
            )
        ),
        reasoning_llm=FixtureStructuredLLM(
            [{"reference_class": "rc", "base_rate": 0.4}, {"rule": "none"}]
        ),
        forecast_llm=FixtureForecastLLM(default_response=0.6),
        supervisor_llm=FixtureSupervisorLLM(),
        leakage_judge=LeakageJudge(FixtureLeakageJudgeLLM()),
        registry_store=store,
    )


@pytest.fixture
def make_service() -> Callable[..., tuple[ForecastService, InMemoryRegistryStore]]:
    def _make(
        *,
        classify: dict = _CLASSIFY,
        normalize: dict = _NORMALIZE,
        providers: tuple[str, ...] = ("anthropic",),
        price_per_call: float | None = None,
        request_log: MutableSequence[str] | None = None,
    ) -> tuple[ForecastService, InMemoryRegistryStore]:
        store = InMemoryRegistryStore()
        forecaster = _forecaster(store, classify=classify)
        conductor = HeuristicConductor(forecaster=forecaster)
        # The intake surface gets its own fixture LLM queue so classify/formalize
        # calls never consume the forecaster's queued responses.
        intake = IntakeService(llm=FixtureStructuredLLM([classify, normalize]), store=store)
        service = ForecastService(
            forecaster=forecaster,
            conductor=conductor,
            store=store,
            intake=intake,
            providers=providers,
            price_per_call=price_per_call,
            request_log=request_log,
        )
        return service, store

    return _make


@pytest.fixture
def make_app(
    make_service: Callable[..., tuple[ForecastService, InMemoryRegistryStore]],
) -> Callable[..., DelphiApp]:
    def _make(**kwargs: object) -> DelphiApp:
        service, _store = make_service(**kwargs)  # type: ignore[arg-type]
        return DelphiApp(service)

    return _make
