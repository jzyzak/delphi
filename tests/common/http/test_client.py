"""Unit tests for the HTTP transport (§8). Network mocked via httpx.MockTransport."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
import pytest

from common.http import (
    HttpClient,
    HttpConfig,
    HttpError,
    HttpNotFound,
    HttpRateLimited,
    HttpTransportFailure,
)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    **cfg: object,
) -> HttpClient:
    config = HttpConfig(retry_backoff_base=0.0, retry_backoff_max=0.0, **cfg)  # type: ignore[arg-type]
    transport = httpx.MockTransport(handler)
    return HttpClient(config=config, client=httpx.Client(transport=transport))


def test_get_json_happy_path() -> None:
    client = _client(lambda _req: httpx.Response(200, json={"a": 1}))
    assert client.get_json("https://x/test") == {"a": 1}


def test_get_text_happy_path() -> None:
    client = _client(lambda _req: httpx.Response(200, text="hello"))
    assert client.get_text("https://x/test") == "hello"


def test_params_and_headers_and_user_agent() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        seen["user_agent"] = req.headers.get("user-agent")
        seen["custom"] = req.headers.get("x-custom")
        return httpx.Response(200, json={})

    client = _client(handler, user_agent="DELPHI test@example.com")
    client.get_json("https://x/y", params={"q": "elections"}, headers={"X-Custom": "v"})
    assert seen["params"] == {"q": "elections"}
    assert seen["user_agent"] == "DELPHI test@example.com"
    assert seen["custom"] == "v"


def test_429_then_success_retries() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    client = _client(handler, max_retries=3)
    assert client.get_json("https://x/y") == {"ok": True}
    assert calls["n"] == 2


def test_500_then_success_retries() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, json={})

    client = _client(handler, max_retries=3)
    client.get_json("https://x/y")
    assert calls["n"] == 2


def test_429_exhausts_and_raises_rate_limited() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    client = _client(handler, max_retries=2)
    with pytest.raises(HttpRateLimited):
        client.get_json("https://x/y")
    assert calls["n"] == 2


def test_404_not_retried() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    client = _client(handler, max_retries=3)
    with pytest.raises(HttpNotFound):
        client.get_json("https://x/y")
    assert calls["n"] == 1


def test_400_raises_http_error_not_retried() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad")

    client = _client(handler, max_retries=3)
    with pytest.raises(HttpError):
        client.get_json("https://x/y")
    assert calls["n"] == 1


def test_invalid_json_raises_http_error() -> None:
    client = _client(lambda _req: httpx.Response(200, text="not json"))
    with pytest.raises(HttpError):
        client.get_json("https://x/y")


def test_transport_error_retried_then_wrapped_as_http_error() -> None:
    # A terminal network failure must surface inside the HttpError taxonomy —
    # a raw httpx exception escaping here crashed a full ablation run when one
    # GDELT SSL handshake timed out (2026-07-15).
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    client = _client(handler, max_retries=2)
    with pytest.raises(HttpTransportFailure, match="transport failure") as excinfo:
        client.get_json("https://x/y")
    assert calls["n"] == 2
    assert isinstance(excinfo.value, HttpError)


def test_connect_timeout_wrapped_as_http_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("_ssl.c:983: The handshake operation timed out")

    client = _client(handler, max_retries=1)
    with pytest.raises(HttpTransportFailure, match="ConnectTimeout"):
        client.get_json("https://x/y")


def test_post_json_sends_body_and_parses_response() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        seen["method"] = req.method
        seen["body"] = _json.loads(req.content.decode("utf-8"))
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"results": []})

    client = _client(handler)
    out = client.post_json(
        "https://x/search", json={"query": "elections"}, headers={"Authorization": "Bearer k"}
    )
    assert out == {"results": []}
    assert seen["method"] == "POST"
    assert seen["body"] == {"query": "elections"}
    assert seen["auth"] == "Bearer k"


def test_post_json_retries_on_429() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] == 1 else httpx.Response(200, json={"ok": True})

    client = _client(handler, max_retries=3)
    assert client.post_json("https://x/y", json={}) == {"ok": True}
    assert calls["n"] == 2


def test_post_json_invalid_json_raises_http_error() -> None:
    client = _client(lambda _req: httpx.Response(200, text="not json"))
    with pytest.raises(HttpError):
        client.post_json("https://x/y", json={})


def test_rate_gate_spaces_requests() -> None:
    client = _client(lambda _req: httpx.Response(200, json={}), min_interval_s=0.05)
    start = time.monotonic()
    client.get_json("https://x/y")
    client.get_json("https://x/y")
    assert time.monotonic() - start >= 0.045
