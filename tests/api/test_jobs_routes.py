"""HTTP tests for the async job surface (submit + status + auth + WSGI).

The job routes exist because hosted front-ends (App Runner) hard-cap request
time below a real forecast's duration; these tests prove the contract the
dashboard relies on: 202-and-poll, idempotent resubmits, and long-poll
plumbing — all hermetic (fixture LLMs, inline or deferred executors).
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

from api.jobs import InMemoryJobStore, JobManager
from api.routes import ForecastService
from api.server import DelphiApp, forecast_runner, wsgi_application
from core.registry.store import InMemoryRegistryStore
from tests.api.test_jobs import DeferredExecutor, InlineExecutor

AS_OF = "2024-06-01T00:00:00+00:00"
TOKEN = "s3cret-token"

MakeService = Callable[..., tuple[ForecastService, InMemoryRegistryStore]]


def _app_with_jobs(
    make_service: MakeService,
    *,
    executor: Any | None = None,
    auth_token: str | None = None,
    **service_kwargs: Any,
) -> tuple[DelphiApp, InMemoryJobStore]:
    service, _store = make_service(**service_kwargs)
    job_store = InMemoryJobStore()
    manager = JobManager(
        store=job_store,
        runner=forecast_runner(service),
        executor=executor if executor is not None else InlineExecutor(),
    )
    return DelphiApp(service, auth_token=auth_token, jobs=manager), job_store


def _submit(app: DelphiApp, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    payload = {"question": "Will X ship?", "as_of": AS_OF}
    if body is not None:
        payload = body
    return app.handle("POST", "/v1/forecast/jobs", payload)


def test_submit_returns_202_with_job_resource(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service, executor=DeferredExecutor())
    status, body = _submit(app)
    assert status == 202
    assert body["object"] == "forecast.job"
    assert body["id"].startswith("fj-")
    assert body["status"] == "queued"
    assert body["result"] is None
    assert body["request"]["question"] == "Will X ship?"


def test_submit_alias_path(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service, executor=DeferredExecutor())
    status, _ = app.handle("POST", "/forecast/jobs", {"question": "q", "as_of": AS_OF})
    assert status == 202


def test_inline_execution_yields_succeeded_job_with_forecast(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)  # inline executor: runs during submit
    status, body = _submit(app)
    assert status == 202
    assert body["status"] == "succeeded"
    result = body["result"]
    assert result["object"] == "forecast.completion"
    assert result["delphi"]["probability"] is not None


def test_get_returns_job_status_and_result(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)
    _, submitted = _submit(app)
    status, body = app.handle("GET", f"/v1/forecast/jobs/{submitted['id']}")
    assert status == 200
    assert body["id"] == submitted["id"]
    assert body["status"] == "succeeded"
    assert body["result"]["delphi"]["probability"] is not None
    assert body["error"] is None


def test_get_alias_path(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)
    _, submitted = _submit(app)
    status, _ = app.handle("GET", f"/forecast/jobs/{submitted['id']}")
    assert status == 200


def test_idempotent_resubmit_returns_200_and_same_job(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service)
    body = {"question": "Will X ship?", "as_of": AS_OF, "idempotency_key": "dash-1"}
    first_status, first = _submit(app, body)
    second_status, second = _submit(app, body)
    assert first_status == 202
    assert second_status == 200
    assert second["id"] == first["id"]
    assert len(store) == 1


def test_refused_question_is_a_succeeded_job_with_refusal(make_service: MakeService) -> None:
    """Refusal is a product answer (§10), not a job failure."""
    app, _ = _app_with_jobs(make_service, classify={"question_type": "unknown"})
    status, body = _submit(app, {"question": "gibberish?", "as_of": AS_OF})
    assert status == 202
    assert body["status"] == "succeeded"
    assert body["result"]["delphi"]["refused"] is True


def test_submit_missing_as_of_creates_no_job(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service)
    status, body = _submit(app, {"question": "q"})
    assert status == 400
    assert body["error"] == "invalid_request"
    assert len(store) == 0


def test_submit_missing_question_creates_no_job(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service)
    status, body = _submit(app, {"as_of": AS_OF})
    assert status == 400
    assert body["error"] == "invalid_request"
    assert len(store) == 0


def test_submit_bad_as_of_creates_no_job(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service)
    status, _ = _submit(app, {"question": "q", "as_of": "not-a-date"})
    assert status == 400
    assert len(store) == 0


def test_submit_provider_opt_out_creates_no_job(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service)
    status, body = _submit(
        app, {"question": "q", "as_of": AS_OF, "provider_opt_out": ["anthropic"]}
    )
    assert status == 403
    assert body["error"] == "provider_opt_out"
    assert len(store) == 0


def test_submit_empty_idempotency_key_is_rejected(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service)
    status, _ = _submit(app, {"question": "q", "as_of": AS_OF, "idempotency_key": ""})
    assert status == 400
    assert len(store) == 0


def test_get_unknown_job_returns_404(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)
    status, body = app.handle("GET", "/v1/forecast/jobs/fj-missing")
    assert status == 404
    assert body["error"] == "not_found"


def test_get_with_invalid_wait_returns_400(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)
    _, submitted = _submit(app)
    for bad in ("soon", "nan", "inf", "-inf"):
        status, body = app.handle(
            "GET", f"/v1/forecast/jobs/{submitted['id']}", query={"wait": bad}
        )
        assert status == 400, bad
        assert body["error"] == "invalid_request"


def test_get_with_wait_on_finished_job_returns_immediately(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)
    _, submitted = _submit(app)
    status, body = app.handle("GET", f"/v1/forecast/jobs/{submitted['id']}", query={"wait": "30"})
    assert status == 200
    assert body["status"] == "succeeded"


def test_job_paths_reject_trailing_slash_and_nesting(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service)
    assert app.handle("GET", "/v1/forecast/jobs/")[0] == 404
    assert app.handle("GET", "/v1/forecast/jobs/a/b")[0] == 404


def test_job_submit_requires_auth(make_service: MakeService) -> None:
    app, store = _app_with_jobs(make_service, auth_token=TOKEN)
    status, body = _submit(app)
    assert status == 401
    assert body["error"] == "unauthorized"
    assert len(store) == 0


def test_job_status_requires_auth(make_service: MakeService) -> None:
    """The result embeds the paid forecast; it must be token-gated too."""
    app, _ = _app_with_jobs(make_service, auth_token=TOKEN)
    status, body = app.handle("GET", "/v1/forecast/jobs/fj-1")
    assert status == 401
    assert body["error"] == "unauthorized"


def test_job_routes_accept_correct_token(make_service: MakeService) -> None:
    app, _ = _app_with_jobs(make_service, auth_token=TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    status, submitted = app.handle(
        "POST", "/v1/forecast/jobs", {"question": "q", "as_of": AS_OF}, headers=headers
    )
    assert status == 202
    status, _ = app.handle("GET", f"/v1/forecast/jobs/{submitted['id']}", headers=headers)
    assert status == 200


def test_default_app_builds_working_job_manager(make_service: MakeService) -> None:
    """DelphiApp with no explicit manager serves jobs end-to-end (real threads)."""
    service, _ = make_service()
    app = DelphiApp(service)
    status, submitted = _submit(app)
    assert status == 202
    status, body = app.handle("GET", f"/v1/forecast/jobs/{submitted['id']}", query={"wait": "10"})
    assert status == 200
    assert body["status"] == "succeeded"
    assert body["result"]["delphi"]["probability"] is not None


class TestWsgiPlumbing:
    def _call(
        self,
        app: Callable[..., list[bytes]],
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query_string: str = "",
    ) -> tuple[str, dict[str, Any]]:
        payload = b"" if body is None else json.dumps(body).encode()
        environ: dict[str, Any] = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": query_string,
            "CONTENT_LENGTH": str(len(payload)),
            "wsgi.input": io.BytesIO(payload),
        }
        captured: dict[str, str] = {}

        def start_response(status: str, _headers: list[tuple[str, str]]) -> None:
            captured["status"] = status

        chunks = app(environ, start_response)
        return captured["status"], json.loads(b"".join(chunks) or b"{}")

    def test_submit_and_poll_through_wsgi(self, make_service: MakeService) -> None:
        app, _ = _app_with_jobs(make_service)
        wsgi = wsgi_application(app)
        status, submitted = self._call(
            wsgi,
            "POST",
            "/v1/forecast/jobs",
            body={"question": "q", "as_of": AS_OF},
        )
        assert status == "202 Accepted"
        status, body = self._call(
            wsgi,
            "GET",
            f"/v1/forecast/jobs/{submitted['id']}",
            query_string="wait=0",
        )
        assert status == "200 OK"
        assert body["status"] == "succeeded"

    def test_query_string_absent_is_fine(self, make_service: MakeService) -> None:
        app, _ = _app_with_jobs(make_service)
        wsgi = wsgi_application(app)
        _, submitted = self._call(
            wsgi, "POST", "/v1/forecast/jobs", body={"question": "q", "as_of": AS_OF}
        )
        environ: dict[str, Any] = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": f"/v1/forecast/jobs/{submitted['id']}",
            "wsgi.input": io.BytesIO(b""),
        }
        captured: dict[str, str] = {}

        def start_response(status: str, _headers: list[tuple[str, str]]) -> None:
            captured["status"] = status

        body = json.loads(b"".join(wsgi(environ, start_response)))
        assert captured["status"] == "200 OK"
        assert body["status"] == "succeeded"

    def test_invalid_wait_through_wsgi(self, make_service: MakeService) -> None:
        app, _ = _app_with_jobs(make_service)
        wsgi = wsgi_application(app)
        _, submitted = self._call(
            wsgi, "POST", "/v1/forecast/jobs", body={"question": "q", "as_of": AS_OF}
        )
        status, body = self._call(
            wsgi,
            "GET",
            f"/v1/forecast/jobs/{submitted['id']}",
            query_string="wait=never",
        )
        assert status.startswith("400")
        assert body["error"] == "invalid_request"
