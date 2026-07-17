"""Tavily search provider adapter (concrete hosted provider).

Tavily is a POST-JSON search API tuned for LLM consumption. This adapter maps
its request/response onto the provider-agnostic contract used by the rest of the
sources layer (:class:`HostedSearchResult`: ``{title, url, content,
published_date, score}``), so the as-of filter, snapshot store, and forecast
chain are unchanged. The API key is resolved at call time from
:mod:`common.secrets` (logical name ``tavily-api-key`` -> env
``DELPHI_SECRET_TAVILY_API_KEY``), never hardcoded (CLAUDE.md §7).

Freshness note: Tavily only returns ``published_date`` for the ``news`` topic.
The as-of filter (Prime Directive §2.1) drops undated results as unsafe, so
``topic="news"`` is the default here to preserve as-of-auditable evidence; use
``topic="general"`` only when you accept that dateless hits will be filtered out.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from common.http.client import HttpClient
from common.secrets import SecretProvider
from sources.providers.hosted import (
    HostedSearchClient,
    HostedSearchConfig,
    HostedSearchResponse,
    HostedSearchResult,
    as_of_date_bound,
)

__all__ = [
    "TAVILY_API_KEY_SECRET",
    "TavilySearchClient",
    "tavily_config",
]

TAVILY_API_KEY_SECRET = "tavily-api-key"
_TAVILY_URL = "https://api.tavily.com/search"


def tavily_config(
    *,
    base_url: str = _TAVILY_URL,
    # v2: queries carry a server-side ``end_date`` as-of bound. The version is
    # part of the snapshot key, so bumping it retires v1 snapshots retrieved
    # against the live (as-of-blind) index — their ranking saw the future.
    version: str = "v2",
    search_depth: str = "advanced",
    topic: str = "news",
    api_key_secret: str = TAVILY_API_KEY_SECRET,
) -> HostedSearchConfig:
    """Build a :class:`HostedSearchConfig` targeting Tavily.

    ``search_depth`` and ``topic`` are carried in ``extra_params`` and merged
    into the POST body by :class:`TavilySearchClient`.
    """
    return HostedSearchConfig(
        base_url=base_url,
        provider="tavily",
        version=version,
        api_key_secret=api_key_secret,
        api_key_header="Authorization",
        query_param="query",
        extra_params={"search_depth": search_depth, "topic": topic},
    )


class TavilySearchClient(HostedSearchClient):
    """Hosted client speaking Tavily's POST ``/search`` contract.

    Overrides :meth:`search` to issue a single POST (Tavily returns up to
    ``max_results`` in one call) and maps the response onto the shared
    :class:`HostedSearchResult` contract. Auth is ``Authorization: Bearer <key>``.
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        config: HostedSearchConfig | None = None,
        secrets: SecretProvider | None = None,
    ) -> None:
        super().__init__(http=http, config=config or tavily_config(), secrets=secrets)

    def _bearer_headers(self) -> dict[str, str] | None:
        if self._config.api_key_secret and self._secrets is not None:
            key = self._secrets.get_secret(self._config.api_key_secret)
            return {self._config.api_key_header: f"Bearer {key}"}
        return None

    def search(
        self, query: str, *, max_results: int = 10, as_of: datetime | None = None
    ) -> HostedSearchResponse:
        if max_results < 1:
            msg = "max_results must be >= 1."
            raise ValueError(msg)
        body: dict[str, Any] = {
            self._config.query_param: query,
            "max_results": max_results,
            **self._config.extra_params,
        }
        if as_of is not None:
            # Server-side as-of bound (§2.1): restrict retrieval AND ranking to
            # publications on or before the ceiling, so post-as-of events cannot
            # shape which pre-as-of articles surface. Day-granular — the exact
            # timestamp is still enforced downstream by ``filter_as_of``.
            body["end_date"] = as_of_date_bound(as_of)
        payload = self._http.post_json(
            self._config.base_url, json=body, headers=self._bearer_headers()
        )
        raw = payload if isinstance(payload, dict) else {"results": []}
        raw_results = raw.get("results", [])
        results = raw_results if isinstance(raw_results, list) else []
        collected = [HostedSearchResult.from_raw(r) for r in results[:max_results]]
        return HostedSearchResponse(
            query=query,
            results=tuple(collected),
            raw={"pages": [raw]},
        )
