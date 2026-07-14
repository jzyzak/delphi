"""Server app + `delphi serve` (C10.5).

A dependency-light HTTP app (no web framework — CLAUDE.md §7/§10 forbids
premature heavyweight infra). ``DelphiApp.handle`` is a pure ``(method, path,
body, headers) -> (status, json)`` dispatcher covering the forecast route plus
health and readiness endpoints, so it is fully testable without binding a
socket. ``wsgi_application`` adapts it to the WSGI protocol (used by both the
stdlib dev server ``serve`` and a production server such as gunicorn via
``api.wsgi:application``).

Auth: when a ``DelphiApp`` is built with an ``auth_token`` the forecast routes
require an ``Authorization: Bearer <token>`` header (constant-time compared);
the health/readiness routes stay open so load balancers can probe them. When
``auth_token`` is ``None`` the endpoint is open (local development).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from http import HTTPStatus
from secrets import compare_digest
from typing import Any

from pydantic import ValidationError

from api.compliance import ProviderOptOutError
from api.routes import ForecastService
from api.schema import ForecastAPIRequest

__all__ = ["DelphiApp", "serve", "wsgi_application"]

_FORECAST_PATHS = frozenset({"/v1/forecast", "/forecast"})

Response = tuple[int, dict[str, Any]]
WSGIApplication = Callable[[dict[str, Any], Any], list[bytes]]


def _bearer_token(headers: Mapping[str, str]) -> str | None:
    """Extract the bearer token from a case-insensitive header mapping."""
    raw = headers.get("authorization")
    if raw is None:
        return None
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


class DelphiApp:
    """Framework-agnostic request dispatcher for the published API."""

    def __init__(self, service: ForecastService, *, auth_token: str | None = None) -> None:
        self._service = service
        self._auth_token = auth_token

    def handle(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Dispatch one request; returns ``(status_code, json_body)``."""
        if method == "GET" and path == "/healthz":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/readyz":
            return 200, {"status": "ready"}
        if method == "POST" and path in _FORECAST_PATHS:
            denied = self._authorize(headers or {})
            if denied is not None:
                return denied
            return self._forecast(body or {})
        return 404, {"error": "not_found", "path": path}

    def _authorize(self, headers: Mapping[str, str]) -> Response | None:
        """Return a 401 response if a required bearer token is missing/invalid."""
        if self._auth_token is None:
            return None
        provided = _bearer_token({key.lower(): value for key, value in headers.items()})
        if provided is None:
            return 401, {"error": "unauthorized", "detail": "missing or malformed bearer token"}
        if not compare_digest(provided, self._auth_token):
            return 401, {"error": "unauthorized", "detail": "invalid bearer token"}
        return None

    def _forecast(self, body: dict[str, Any]) -> Response:
        try:
            request = ForecastAPIRequest.model_validate(body)
        except ValidationError as exc:
            return 400, {"error": "invalid_request", "detail": exc.errors(include_url=False)}
        try:
            response = self._service.forecast(request)
        except ProviderOptOutError as exc:
            return 403, {"error": "provider_opt_out", "detail": str(exc)}
        except ValueError as exc:
            return 400, {"error": "invalid_request", "detail": str(exc)}
        return 200, response.model_dump(mode="json")


def wsgi_application(app: DelphiApp) -> WSGIApplication:
    """Adapt a ``DelphiApp`` into a WSGI callable (dev server + gunicorn)."""

    def wsgi(environ: dict[str, Any], start_response: Any) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        headers: dict[str, str] = {}
        authorization = environ.get("HTTP_AUTHORIZATION")
        if authorization is not None:
            headers["authorization"] = authorization

        body: dict[str, Any] = {}
        if method == "POST":
            try:
                size = int(environ.get("CONTENT_LENGTH") or 0)
            except ValueError:
                size = 0
            raw = environ["wsgi.input"].read(size) if size else b""
            if raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    return _bad_request(start_response, "invalid JSON body")
                if not isinstance(parsed, dict):
                    return _bad_request(start_response, "body must be a JSON object")
                body = parsed

        status, payload = app.handle(method, path, body, headers=headers)
        return _respond(start_response, status, payload)

    return wsgi


def _bad_request(start_response: Any, detail: str) -> list[bytes]:
    return _respond(start_response, 400, {"error": "invalid_request", "detail": detail})


def _respond(start_response: Any, status: int, payload: dict[str, Any]) -> list[bytes]:
    data = json.dumps(payload).encode("utf-8")
    try:
        reason = HTTPStatus(status).phrase
    except ValueError:  # pragma: no cover - all statuses we emit are valid
        reason = ""
    start_response(
        f"{status} {reason}".strip(),
        [("Content-Type", "application/json"), ("Content-Length", str(len(data)))],
    )
    return [data]


def serve(
    app: DelphiApp, *, host: str = "127.0.0.1", port: int = 8080
) -> None:  # pragma: no cover - binds a socket
    """Run ``app`` over a stdlib WSGI server (local development entry point)."""
    from wsgiref.simple_server import make_server

    with make_server(host, port, wsgi_application(app)) as httpd:
        httpd.serve_forever()
