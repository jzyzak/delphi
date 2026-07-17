"""Server app + `delphi serve` (C10.5).

A dependency-light HTTP app (no web framework — CLAUDE.md §7/§10 forbids
premature heavyweight infra). ``DelphiApp.handle`` is a pure ``(method, path,
body, headers, query) -> (status, json)`` dispatcher covering the forecast +
intake + async-job routes plus health and readiness endpoints, so it is fully
testable without binding a socket. ``wsgi_application`` adapts it to the WSGI
protocol (used by both the stdlib dev server ``serve`` and a production server
such as gunicorn via ``api.wsgi:application``).

Auth: when a ``DelphiApp`` is built with an ``auth_token`` the forecast,
intake, and job routes require an ``Authorization: Bearer <token>`` header
(constant-time compared); the health/readiness routes stay open so load
balancers can probe them. When ``auth_token`` is ``None`` the endpoint is open
(local development).

Async jobs: hosted front-ends cap total request time (App Runner: a hard,
non-configurable 120s), which a real forecast exceeds. ``POST
/v1/forecast/jobs`` enqueues and returns 202 immediately; ``GET
/v1/forecast/jobs/{id}?wait=N`` long-polls the status/result. See ``api.jobs``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from http import HTTPStatus
from secrets import compare_digest
from typing import Any, TypeVar
from urllib.parse import parse_qsl

from pydantic import BaseModel, ValidationError

from api.compliance import ProviderOptOutError
from api.jobs import InMemoryJobStore, JobManager, JobRunner
from api.routes import ForecastService
from api.schema import (
    ForecastAPIRequest,
    ForecastJobSubmitRequest,
    IntakeAPIRequest,
    build_job_response,
)

__all__ = ["DelphiApp", "forecast_runner", "serve", "wsgi_application"]

_FORECAST_PATHS = frozenset({"/v1/forecast", "/forecast"})
_CLASSIFY_PATHS = frozenset({"/v1/classify", "/classify"})
_FORMALIZE_PATHS = frozenset({"/v1/formalize", "/formalize"})
_JOB_SUBMIT_PATHS = frozenset({"/v1/forecast/jobs", "/forecast/jobs"})
_JOB_STATUS_PREFIXES = ("/v1/forecast/jobs/", "/forecast/jobs/")

Response = tuple[int, dict[str, Any]]
WSGIApplication = Callable[[dict[str, Any], Any], list[bytes]]

_RequestT = TypeVar("_RequestT", bound=BaseModel)


def forecast_runner(service: ForecastService) -> JobRunner:
    """Adapt ``service.forecast`` into the job runner (payload dict -> response).

    Validation happens here (not in ``api.jobs``) so the jobs module stays
    schema-agnostic; a payload that fails validation fails the job with an
    ``invalid_request`` error (submit-time prechecks make that unlikely).
    """

    def run(payload: Mapping[str, Any]) -> BaseModel:
        return service.forecast(ForecastAPIRequest.model_validate(dict(payload)))

    return run


def _job_path_id(path: str) -> str | None:
    """Extract the job id from a status path, or ``None`` if not a job path."""
    for prefix in _JOB_STATUS_PREFIXES:
        if path.startswith(prefix):
            job_id = path[len(prefix) :]
            if job_id and "/" not in job_id:
                return job_id
    return None


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

    def __init__(
        self,
        service: ForecastService,
        *,
        auth_token: str | None = None,
        jobs: JobManager | None = None,
    ) -> None:
        self._service = service
        self._auth_token = auth_token
        # Default: in-memory jobs (correct for the single-process dev server
        # and tests). Production wiring passes a Postgres-backed manager so
        # polls landing on any worker/instance see every job.
        self._jobs = (
            jobs
            if jobs is not None
            else JobManager(store=InMemoryJobStore(), runner=forecast_runner(service))
        )

    @property
    def auth_enabled(self) -> bool:
        """True when forecast routes require a bearer token (False = open, dev only)."""
        return self._auth_token is not None

    def handle(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, str] | None = None,
    ) -> Response:
        """Dispatch one request; returns ``(status_code, json_body)``."""
        if method == "GET" and path == "/healthz":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/readyz":
            return 200, {"status": "ready"}
        if method == "GET":
            job_id = _job_path_id(path)
            if job_id is not None:
                denied = self._authorize(headers or {})
                if denied is not None:
                    return denied
                return self._job_status(job_id, query or {})
        if method == "POST":
            handler = self._post_handler(path)
            if handler is not None:
                denied = self._authorize(headers or {})
                if denied is not None:
                    return denied
                return handler(body or {})
        return 404, {"error": "not_found", "path": path}

    def _post_handler(self, path: str) -> Callable[[dict[str, Any]], Response] | None:
        """Map a POST path to its (auth-gated) handler, or ``None`` for 404."""
        if path in _FORECAST_PATHS:
            return self._forecast
        if path in _CLASSIFY_PATHS:
            return self._classify
        if path in _FORMALIZE_PATHS:
            return self._formalize
        if path in _JOB_SUBMIT_PATHS:
            return self._submit_job
        return None

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
        return self._run(body, ForecastAPIRequest, self._service.forecast)

    def _classify(self, body: dict[str, Any]) -> Response:
        return self._run(body, IntakeAPIRequest, self._service.classify)

    def _formalize(self, body: dict[str, Any]) -> Response:
        return self._run(body, IntakeAPIRequest, self._service.formalize)

    def _submit_job(self, body: dict[str, Any]) -> Response:
        """``POST /v1/forecast/jobs``: validate, enqueue, return immediately.

        202 for a newly created job, 200 for an idempotency-key hit (the
        existing job's current state; nothing new is spent). Requests that
        would fail admission are rejected here and never become jobs.
        """
        try:
            request = ForecastJobSubmitRequest.model_validate(body)
        except ValidationError as exc:
            return 400, {"error": "invalid_request", "detail": exc.errors(include_url=False)}
        try:
            self._service.precheck_forecast(request)
        except ProviderOptOutError as exc:
            return 403, {"error": "provider_opt_out", "detail": str(exc)}
        except ValueError as exc:
            return 400, {"error": "invalid_request", "detail": str(exc)}
        job, created = self._jobs.submit(
            request.job_payload(), idempotency_key=request.idempotency_key
        )
        status = 202 if created else 200
        return status, build_job_response(job).model_dump(mode="json")

    def _job_status(self, job_id: str, query: Mapping[str, str]) -> Response:
        """``GET /v1/forecast/jobs/{id}``: status/result, optional long-poll.

        ``?wait=N`` holds the request up to N seconds (clamped server-side)
        until the job is terminal — which also keeps a CPU-throttling host
        (App Runner) running the worker at full speed while the client waits.
        """
        raw_wait = query.get("wait")
        wait_s = 0.0
        if raw_wait is not None:
            try:
                wait_s = float(raw_wait)
            except ValueError:
                wait_s = math.nan
            if not math.isfinite(wait_s):  # rejects non-numeric, 'nan', '±inf'
                return 400, {
                    "error": "invalid_request",
                    "detail": f"wait must be a finite number of seconds, got {raw_wait!r}",
                }
        job = self._jobs.get(job_id, wait_s=wait_s)
        if job is None:
            return 404, {"error": "not_found", "detail": f"no forecast job {job_id!r}"}
        return 200, build_job_response(job).model_dump(mode="json")

    def _run(
        self,
        body: dict[str, Any],
        request_model: type[_RequestT],
        route: Callable[[_RequestT], BaseModel],
    ) -> Response:
        """Validate ``body`` against ``request_model``, run ``route``, map errors."""
        try:
            request = request_model.model_validate(body)
        except ValidationError as exc:
            return 400, {"error": "invalid_request", "detail": exc.errors(include_url=False)}
        try:
            response = route(request)
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
        query = dict(parse_qsl(environ.get("QUERY_STRING") or "", keep_blank_values=True))
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

        status, payload = app.handle(method, path, body, headers=headers, query=query)
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
