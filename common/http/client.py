"""Generic, polite HTTP transport with retries and rate limiting.

Domain-agnostic (CLAUDE.md section 11): knows nothing about forecasts or
sources. Wraps an injectable ``httpx.Client`` so tests can supply an
``httpx.MockTransport``-backed client and never touch the network.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx
import structlog
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.http.config import HttpConfig
from common.http.errors import (
    HttpError,
    HttpNotFound,
    HttpRateLimited,
    HttpTransportFailure,
    _TransientHttpError,
)

__all__ = ["HttpClient"]

_LOG = structlog.get_logger(__name__)

_RETRYABLE = (HttpRateLimited, _TransientHttpError, httpx.TransportError)


class HttpClient:
    """Polite JSON/text HTTP client with bounded retries and a rate gate.

    Construction never touches the network; a real ``httpx.Client`` is created
    lazily on first use unless one is injected. ``user_agent`` from the config
    is merged into every request's headers so it applies regardless of how an
    injected client was constructed.
    """

    def __init__(
        self,
        *,
        config: HttpConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or HttpConfig()
        self._client = client
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    @property
    def config(self) -> HttpConfig:
        return self._config

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._config.request_timeout_s)
        return self._client

    def _merge_headers(self, headers: Mapping[str, str] | None) -> dict[str, str] | None:
        merged: dict[str, str] = {}
        if self._config.user_agent:
            merged["User-Agent"] = self._config.user_agent
        if headers:
            merged.update(headers)
        return merged or None

    def _rate_gate(self) -> None:
        """Block until at least ``min_interval_s`` has elapsed since last call."""
        interval = self._config.min_interval_s
        if interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_request_at + interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_request_at = now

    @staticmethod
    def _check_status(url: str, response: httpx.Response) -> httpx.Response:
        status = response.status_code
        if status == 429:
            raise HttpRateLimited(f"429 Too Many Requests for {url}")
        if status == 404:
            raise HttpNotFound(f"404 Not Found for {url}")
        if 500 <= status < 600:
            raise _TransientHttpError(f"{status} server error for {url}")
        if status >= 400:
            raise HttpError(f"{status} client error for {url}: {response.text[:200]!r}")
        return response

    def _raw_request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        json: Any | None,
    ) -> httpx.Response:
        self._rate_gate()
        client = self._ensure_client()
        response = client.request(
            method, url, params=params, headers=self._merge_headers(headers), json=json
        )
        return self._check_status(url, response)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        json: Any | None = None,
    ) -> httpx.Response:
        cfg = self._config
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(cfg.max_retries),
                wait=wait_exponential(multiplier=cfg.retry_backoff_base, max=cfg.retry_backoff_max),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            ):
                with attempt:
                    try:
                        return self._raw_request(
                            method, url, params=params, headers=headers, json=json
                        )
                    except _RETRYABLE as exc:
                        _LOG.info(
                            "http.retryable_error",
                            url=url,
                            method=method,
                            error_type=type(exc).__name__,
                        )
                        raise
        except httpx.TransportError as exc:
            # Terminal network failure: surface it inside the HttpError taxonomy
            # so callers (e.g. the composite searcher) can skip the provider
            # instead of crashing on a raw httpx exception.
            msg = f"transport failure for {url}: {type(exc).__name__}: {exc}"
            raise HttpTransportFailure(msg) from exc
        raise HttpError("unreachable")  # pragma: no cover - reraise=True

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """GET ``url`` and parse the response body as JSON."""
        response = self._request_with_retry("GET", url, params=params, headers=headers)
        return self._parse_json(url, response)

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        """GET ``url`` and return the response body as text."""
        return self._request_with_retry("GET", url, params=params, headers=headers).text

    def post_json(
        self,
        url: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """POST ``json`` to ``url`` and parse the response body as JSON.

        Shares the rate gate + bounded-retry path with ``get_json`` (429/5xx/
        transport errors retried). Used by POST-based providers such as Tavily.
        """
        response = self._request_with_retry("POST", url, params=params, headers=headers, json=json)
        return self._parse_json(url, response)

    @staticmethod
    def _parse_json(url: str, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            msg = f"response from {url} is not valid JSON"
            raise HttpError(msg) from exc

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
