"""Tests for the HTTP dispatcher (C10.5) — round-trip without binding a socket."""

from __future__ import annotations

from collections.abc import Callable

from api.server import DelphiApp

AS_OF = "2024-06-01T00:00:00+00:00"

MakeApp = Callable[..., DelphiApp]


def test_healthz(make_app: MakeApp) -> None:
    status, body = make_app().handle("GET", "/healthz")
    assert status == 200
    assert body == {"status": "ok"}


def test_readyz(make_app: MakeApp) -> None:
    status, body = make_app().handle("GET", "/readyz")
    assert status == 200
    assert body["status"] == "ready"


def test_forecast_round_trip(make_app: MakeApp) -> None:
    status, body = make_app().handle(
        "POST", "/v1/forecast", {"question": "Will X ship?", "as_of": AS_OF}
    )
    assert status == 200
    assert body["object"] == "forecast.completion"
    assert body["delphi"]["probability"] is not None


def test_forecast_alias_path(make_app: MakeApp) -> None:
    status, _ = make_app().handle("POST", "/forecast", {"question": "q", "as_of": AS_OF})
    assert status == 200


def test_invalid_request_returns_400(make_app: MakeApp) -> None:
    status, body = make_app().handle("POST", "/v1/forecast", {"question": "q"})  # missing as_of
    assert status == 400
    assert body["error"] == "invalid_request"


def test_missing_question_returns_400(make_app: MakeApp) -> None:
    status, body = make_app().handle("POST", "/v1/forecast", {"as_of": AS_OF})
    assert status == 400
    assert body["error"] == "invalid_request"


def test_provider_opt_out_returns_403(make_app: MakeApp) -> None:
    status, body = make_app().handle(
        "POST",
        "/v1/forecast",
        {"question": "q", "as_of": AS_OF, "provider_opt_out": ["anthropic"]},
    )
    assert status == 403
    assert body["error"] == "provider_opt_out"


def test_unknown_route_returns_404(make_app: MakeApp) -> None:
    status, body = make_app().handle("GET", "/nope")
    assert status == 404
    assert body["error"] == "not_found"


def test_post_without_body(make_app: MakeApp) -> None:
    status, _ = make_app().handle("POST", "/v1/forecast", None)
    assert status == 400
