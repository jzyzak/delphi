"""WSGI adapter + production entry point (`api.wsgi`)."""

from __future__ import annotations

import io
import json
from collections.abc import Callable

import pytest

import api.wsgi as wsgi_module
from api.routes import ForecastService
from api.server import DelphiApp, wsgi_application
from api.wsgi import API_TOKEN_ENV, application, build_application
from core.registry.store import InMemoryRegistryStore

AS_OF = "2024-06-01T00:00:00+00:00"
MakeService = Callable[..., tuple[ForecastService, InMemoryRegistryStore]]


def _call(
    app: Callable,
    method: str,
    path: str,
    *,
    body: dict | None = None,
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str], object]:
    payload = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
    environ: dict[str, object] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(payload)),
        "wsgi.input": io.BytesIO(payload),
    }
    environ.update(headers or {})
    captured: dict[str, object] = {}

    def start_response(status: str, response_headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = dict(response_headers)

    chunks = app(environ, start_response)
    parsed = json.loads(b"".join(chunks) or b"{}")
    return captured["status"], captured["headers"], parsed  # type: ignore[return-value]


def test_healthz_status_line_and_body(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, headers, body = _call(app, "GET", "/healthz")
    assert status == "200 OK"
    assert headers["Content-Type"] == "application/json"
    assert body == {"status": "ok"}


def test_forecast_round_trip(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, _, body = _call(app, "POST", "/v1/forecast", body={"question": "q", "as_of": AS_OF})
    assert status == "200 OK"
    assert body["object"] == "forecast.completion"  # type: ignore[index]


def test_classify_round_trip(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, _, body = _call(app, "POST", "/v1/classify", body={"question": "q"})
    assert status == "200 OK"
    assert body["object"] == "question.classification"  # type: ignore[index]


def test_formalize_round_trip(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, _, body = _call(app, "POST", "/v1/formalize", body={"question": "q"})
    assert status == "200 OK"
    assert body["object"] == "question.formalization"  # type: ignore[index]


def test_invalid_json_body_returns_400(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, _, body = _call(app, "POST", "/v1/forecast", raw=b"{not json")
    assert status.startswith("400")
    assert body["error"] == "invalid_request"  # type: ignore[index]


def test_non_object_json_body_returns_400(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, _, _ = _call(app, "POST", "/v1/forecast", raw=b"[1, 2, 3]")
    assert status.startswith("400")


def test_unknown_route_returns_404(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    status, _, body = _call(app, "GET", "/nope")
    assert status.startswith("404")
    assert body["error"] == "not_found"  # type: ignore[index]


def test_malformed_content_length_is_treated_as_empty_body(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service))
    environ: dict[str, object] = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/v1/forecast",
        "CONTENT_LENGTH": "not-a-number",
        "wsgi.input": io.BytesIO(b""),
    }
    captured: dict[str, str] = {}

    def start_response(status: str, _headers: list[tuple[str, str]]) -> None:
        captured["status"] = status

    body = json.loads(b"".join(app(environ, start_response)) or b"{}")
    assert captured["status"].startswith("400")  # empty body -> missing as_of
    assert body["error"] == "invalid_request"


def test_wsgi_enforces_bearer_auth(make_service: MakeService) -> None:
    service, _ = make_service()
    app = wsgi_application(DelphiApp(service, auth_token="tok"))
    denied, _, _ = _call(app, "POST", "/v1/forecast", body={"question": "q", "as_of": AS_OF})
    assert denied.startswith("401")
    ok, _, _ = _call(
        app,
        "POST",
        "/v1/forecast",
        body={"question": "q", "as_of": AS_OF},
        headers={"HTTP_AUTHORIZATION": "Bearer tok"},
    )
    assert ok.startswith("200")


def test_build_application_requires_token(make_service: MakeService) -> None:
    with pytest.raises(RuntimeError, match=API_TOKEN_ENV):
        build_application(environ={})


def test_build_application_wires_token_into_app(make_service: MakeService) -> None:
    service, _ = make_service()
    captured: dict[str, str] = {}

    def factory(*, auth_token: str) -> DelphiApp:
        captured["auth_token"] = auth_token
        return DelphiApp(service, auth_token=auth_token)

    app = build_application(app_factory=factory, environ={API_TOKEN_ENV: "tok"})
    assert captured["auth_token"] == "tok"
    # Fail-closed: no header -> 401; correct header -> 200.
    denied, _, _ = _call(app, "POST", "/v1/forecast", body={"question": "q", "as_of": AS_OF})
    assert denied.startswith("401")
    ok, _, _ = _call(
        app,
        "POST",
        "/v1/forecast",
        body={"question": "q", "as_of": AS_OF},
        headers={"HTTP_AUTHORIZATION": "Bearer tok"},
    )
    assert ok.startswith("200")


def test_application_builds_once_and_delegates(
    make_service: MakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"built": 0}
    recorded: list[str] = []

    def fake_wsgi(environ: dict, start_response: Callable) -> list[bytes]:
        recorded.append(environ["PATH_INFO"])
        start_response("200 OK", [("Content-Type", "application/json")])
        return [b"{}"]

    def fake_build(**_kwargs: object) -> Callable:
        calls["built"] += 1
        return fake_wsgi

    monkeypatch.setattr(wsgi_module, "_cached", None)
    monkeypatch.setattr(wsgi_module, "build_application", fake_build)

    _call(application, "POST", "/v1/forecast", body={"question": "q", "as_of": AS_OF})
    _call(application, "POST", "/v1/forecast", body={"question": "q", "as_of": AS_OF})
    assert calls["built"] == 1  # built once, cached thereafter
    assert recorded == ["/v1/forecast", "/v1/forecast"]


def test_application_health_needs_no_heavy_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**_kwargs: object) -> Callable:
        raise AssertionError("health probes must not build the forecaster graph")

    monkeypatch.setattr(wsgi_module, "_cached", None)
    monkeypatch.setattr(wsgi_module, "build_application", explode)

    status, _, body = _call(application, "GET", "/healthz")
    assert status == "200 OK"
    assert body == {"status": "ok"}
    assert _call(application, "GET", "/readyz")[2] == {"status": "ready"}
