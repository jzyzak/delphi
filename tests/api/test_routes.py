"""Tests for tier routing + envelope completeness (C10.2 / C10.3)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from api.compliance import ProviderOptOutError
from api.routes import ForecastService
from api.schema import ForecastAPIRequest
from core.registry.store import InMemoryRegistryStore

AS_OF = "2024-06-01T00:00:00+00:00"

MakeService = Callable[..., tuple[ForecastService, InMemoryRegistryStore]]


def _request(**kwargs: object) -> ForecastAPIRequest:
    return ForecastAPIRequest(question="Will X ship?", as_of=AS_OF, **kwargs)  # type: ignore[arg-type]


class TestTierRouting:
    def test_shallow_tier(self, make_service: MakeService) -> None:
        service, _ = make_service()
        response = service.forecast(_request(tier="delphi"))
        assert response.usage.tier == "delphi"
        assert response.usage.model_calls == 1
        assert not response.delphi.refused
        assert response.delphi.workflow is None

    def test_deep_tier_includes_workflow_and_red_team(self, make_service: MakeService) -> None:
        service, _ = make_service()
        response = service.forecast(_request(tier="delphi_deep"))
        assert response.usage.model_calls == 2
        assert response.delphi.workflow is not None
        assert response.delphi.red_team_counter


class TestEnvelopeCompleteness:
    def test_all_section9_fields_present(self, make_service: MakeService) -> None:
        service, _ = make_service()
        envelope = service.forecast(_request()).delphi
        assert envelope.probability is not None
        assert envelope.confidence_band is not None
        assert envelope.confidence_band.low <= envelope.probability <= envelope.confidence_band.high
        assert envelope.rationale
        assert envelope.evidence  # provenance present
        assert envelope.evidence[0].knowledge_time.year == 2024
        assert envelope.calibration_metadata
        assert envelope.resolution_criteria
        assert envelope.reproducibility_handle
        assert envelope.providers == ("anthropic",)

    def test_refusal_envelope(self, make_service: MakeService) -> None:
        service, _ = make_service(classify={"question_type": "unknown"})
        envelope = service.forecast(_request()).delphi
        assert envelope.refused
        assert envelope.probability is None


class TestCompliance:
    def test_provider_opt_out_fails_closed(self, make_service: MakeService) -> None:
        service, _ = make_service(providers=("anthropic",))
        with pytest.raises(ProviderOptOutError):
            service.forecast(_request(provider_opt_out=("anthropic",)))

    def test_retention_opt_out_not_logged(self, make_service: MakeService) -> None:
        log: list[str] = []
        service, _ = make_service(request_log=log)
        service.forecast(_request(retention_opt_out=True))
        assert log == []
        assert service.forecast(_request()).delphi.retained  # default retains + logs
        assert log == ["Will X ship?"]

    def test_usage_cost_reported_with_price(self, make_service: MakeService) -> None:
        service, _ = make_service(price_per_call=0.02)
        response = service.forecast(_request(tier="delphi_deep"))
        assert response.usage.cost_usd == pytest.approx(0.04)
