"""Tests for tier routing + intake surfaces + envelope completeness (C10.2 / C10.3)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from api.compliance import ProviderOptOutError
from api.routes import ForecastService
from api.schema import ForecastAPIRequest, IntakeAPIRequest
from core.registry.store import InMemoryRegistryStore

AS_OF = "2024-06-01T00:00:00+00:00"

MakeService = Callable[..., tuple[ForecastService, InMemoryRegistryStore]]


def _request(**kwargs: object) -> ForecastAPIRequest:
    return ForecastAPIRequest(question="Will X ship?", as_of=AS_OF, **kwargs)  # type: ignore[arg-type]


def _intake_request(**kwargs: object) -> IntakeAPIRequest:
    return IntakeAPIRequest(question="Will X ship?", **kwargs)  # type: ignore[arg-type]


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


class TestClassify:
    def test_returns_typed_classification(self, make_service: MakeService) -> None:
        service, _ = make_service()
        response = service.classify(_intake_request())
        assert response.object == "question.classification"
        assert response.classification.question_type == "binary"
        assert response.classification.entities == ("X",)
        assert response.usage.tier == "classify"
        assert response.usage.model_calls == 1

    def test_never_records_to_registry(self, make_service: MakeService) -> None:
        service, store = make_service()
        service.classify(_intake_request())
        assert store.questions_by_domain("tech") == ()

    def test_provider_opt_out_fails_closed(self, make_service: MakeService) -> None:
        service, _ = make_service(providers=("anthropic",))
        with pytest.raises(ProviderOptOutError):
            service.classify(_intake_request(provider_opt_out=("anthropic",)))

    def test_retention_logging(self, make_service: MakeService) -> None:
        log: list[str] = []
        service, _ = make_service(request_log=log)
        service.classify(_intake_request(retention_opt_out=True))
        assert log == []
        service.classify(_intake_request())
        assert log == ["Will X ship?"]

    def test_cost_reported_with_price(self, make_service: MakeService) -> None:
        service, _ = make_service(price_per_call=0.02)
        assert service.classify(_intake_request()).usage.cost_usd == pytest.approx(0.02)


class TestFormalize:
    def test_accepted_returns_resolvable_form(self, make_service: MakeService) -> None:
        service, _ = make_service()
        response = service.formalize(_intake_request())
        assert response.object == "question.formalization"
        assert not response.refused
        assert response.formalized is not None
        assert response.formalized.text == "Will X ship by 2025?"
        assert response.formalized.domain == "tech"
        assert response.formalized.resolution_criteria == "Resolves YES on GA announcement."
        assert response.classification.question_type == "binary"
        assert response.usage.tier == "formalize"
        assert response.usage.model_calls == 2

    def test_never_records_to_registry(self, make_service: MakeService) -> None:
        service, store = make_service()
        response = service.formalize(_intake_request())
        assert not response.refused
        assert store.questions_by_domain("tech") == ()

    def test_unknown_type_refused(self, make_service: MakeService) -> None:
        service, _ = make_service(classify={"question_type": "unknown"})
        response = service.formalize(_intake_request())
        assert response.refused
        assert response.refusal_reason == "unknown_type"
        assert response.formalized is None

    def test_underspecified_refused(self, make_service: MakeService) -> None:
        service, _ = make_service(normalize={"resolution_criteria": ""})
        response = service.formalize(_intake_request())
        assert response.refused
        assert response.refusal_reason == "underspecified"

    def test_as_of_enables_already_resolved_check(self, make_service: MakeService) -> None:
        # The fixture close_time is 2025-06-01; an as-of after it must refuse.
        service, _ = make_service()
        response = service.formalize(_intake_request(as_of="2026-01-01T00:00:00+00:00"))
        assert response.refused
        assert response.refusal_reason == "already_resolved"

    def test_provider_opt_out_fails_closed(self, make_service: MakeService) -> None:
        service, _ = make_service(providers=("anthropic",))
        with pytest.raises(ProviderOptOutError):
            service.formalize(_intake_request(provider_opt_out=("anthropic",)))


class TestPrecheckForecast:
    """Submit-time admission for async jobs: same errors, zero side effects."""

    def test_valid_request_passes(self, make_service: MakeService) -> None:
        service, _ = make_service()
        service.precheck_forecast(_request())  # must not raise

    def test_provider_opt_out_raises(self, make_service: MakeService) -> None:
        service, _ = make_service(providers=("anthropic",))
        with pytest.raises(ProviderOptOutError):
            service.precheck_forecast(_request(provider_opt_out=("anthropic",)))

    def test_missing_question_raises(self, make_service: MakeService) -> None:
        service, _ = make_service()
        with pytest.raises(ValueError, match="no question"):
            service.precheck_forecast(ForecastAPIRequest(as_of=AS_OF))

    def test_bad_as_of_raises(self, make_service: MakeService) -> None:
        service, _ = make_service()
        with pytest.raises(ValueError):
            service.precheck_forecast(ForecastAPIRequest(question="q", as_of="not-a-date"))

    def test_no_retention_logging_side_effect(self, make_service: MakeService) -> None:
        """Precheck must not log: retention logging belongs to execution."""
        log: list[str] = []
        service, _ = make_service(request_log=log)
        service.precheck_forecast(_request())
        assert log == []

    def test_no_registry_writes(self, make_service: MakeService) -> None:
        service, store = make_service()
        service.precheck_forecast(_request())
        assert store.questions_by_domain("tech") == ()


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
