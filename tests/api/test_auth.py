"""Bearer-token auth on the published API (fail-closed forecast routes)."""

from __future__ import annotations

from collections.abc import Callable

from api.routes import ForecastService
from api.server import DelphiApp
from core.registry.store import InMemoryRegistryStore

AS_OF = "2024-06-01T00:00:00+00:00"
TOKEN = "s3cret-token"

MakeService = Callable[..., tuple[ForecastService, InMemoryRegistryStore]]


def _forecast(app: DelphiApp, headers: dict[str, str] | None = None) -> tuple[int, dict]:
    return app.handle("POST", "/v1/forecast", {"question": "q", "as_of": AS_OF}, headers=headers)


def test_open_when_no_token_configured(make_service: MakeService) -> None:
    service, _ = make_service()
    status, body = _forecast(DelphiApp(service))
    assert status == 200
    assert body["object"] == "forecast.completion"


def test_missing_token_is_rejected(make_service: MakeService) -> None:
    service, _ = make_service()
    status, body = _forecast(DelphiApp(service, auth_token=TOKEN))
    assert status == 401
    assert body["error"] == "unauthorized"


def test_wrong_token_is_rejected(make_service: MakeService) -> None:
    service, _ = make_service()
    app = DelphiApp(service, auth_token=TOKEN)
    status, body = _forecast(app, {"Authorization": "Bearer nope"})
    assert status == 401
    assert body["error"] == "unauthorized"


def test_malformed_scheme_is_rejected(make_service: MakeService) -> None:
    service, _ = make_service()
    app = DelphiApp(service, auth_token=TOKEN)
    status, _ = _forecast(app, {"Authorization": TOKEN})  # no "Bearer " scheme
    assert status == 401


def test_correct_token_is_accepted(make_service: MakeService) -> None:
    service, _ = make_service()
    app = DelphiApp(service, auth_token=TOKEN)
    status, body = _forecast(app, {"Authorization": f"Bearer {TOKEN}"})
    assert status == 200
    assert body["delphi"]["probability"] is not None


def test_header_name_is_case_insensitive(make_service: MakeService) -> None:
    service, _ = make_service()
    app = DelphiApp(service, auth_token=TOKEN)
    status, _ = _forecast(app, {"authorization": f"bearer {TOKEN}"})
    assert status == 200


def test_health_endpoints_stay_open_with_auth(make_service: MakeService) -> None:
    service, _ = make_service()
    app = DelphiApp(service, auth_token=TOKEN)
    assert app.handle("GET", "/healthz")[0] == 200
    assert app.handle("GET", "/readyz")[0] == 200
