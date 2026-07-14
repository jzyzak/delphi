"""Production WSGI entry point (gunicorn target ``api.wsgi:application``).

Unlike ``delphi serve`` (a convenience dev server), this module is what a real
process manager imports. It is **fail-closed on auth**: it refuses to build the
application unless ``DELPHI_SECRET_API_TOKEN`` is set, so a deployed endpoint
can never accidentally be left open (CLAUDE.md §9 compliance surface).

The real forecaster graph (Postgres + LLM + hosted search) is built lazily on
the first request, so importing this module is cheap and side-effect free
(hermetic tests import it without any infra).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from api.server import DelphiApp, WSGIApplication, wsgi_application

__all__ = ["API_TOKEN_ENV", "application", "build_application"]

API_TOKEN_ENV = "DELPHI_SECRET_API_TOKEN"

# Static health/readiness payloads, answered without building the heavy
# forecaster graph so load-balancer probes work before (and independently of)
# Postgres/LLM connectivity.
_HEALTH: dict[str, dict[str, str]] = {
    "/healthz": {"status": "ok"},
    "/readyz": {"status": "ready"},
}

AppFactory = Callable[..., DelphiApp]

_cached: WSGIApplication | None = None


def _production_app_factory(*, auth_token: str) -> DelphiApp:  # pragma: no cover - heavy wiring
    """Build the real Postgres/LLM-backed app with bearer auth enforced."""
    from common.cli import _default_api_app

    return _default_api_app(auth_token=auth_token)


def build_application(
    *,
    app_factory: AppFactory = _production_app_factory,
    environ: dict[str, str] | None = None,
) -> WSGIApplication:
    """Build the WSGI application, requiring a bearer token (fail-closed)."""
    env = environ if environ is not None else os.environ
    token = env.get(API_TOKEN_ENV)
    if not token:
        msg = (
            f"{API_TOKEN_ENV} must be set to serve the API "
            "(fail-closed bearer auth; refusing to start an open endpoint)."
        )
        raise RuntimeError(msg)
    return wsgi_application(app_factory(auth_token=token))


def application(environ: dict[str, Any], start_response: Any) -> list[bytes]:
    """WSGI entry point; builds the real app once on the first forecast request.

    Health/readiness probes are answered statically (no Postgres/LLM needed).
    """
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    if method == "GET" and path in _HEALTH:
        data = json.dumps(_HEALTH[path]).encode("utf-8")
        start_response(
            "200 OK",
            [("Content-Type", "application/json"), ("Content-Length", str(len(data)))],
        )
        return [data]

    global _cached
    if _cached is None:
        _cached = build_application()
    return _cached(environ, start_response)
