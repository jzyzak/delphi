"""Hosted web-search provider client (C3.1).

Wraps the polite :class:`~common.http.client.HttpClient` to call a hosted search
API and parse its JSON into typed results. Provider-agnostic by design: the
client targets a small, documented JSON contract — a top-level ``results`` array
of ``{title, url, content, published_date, score}`` objects — so any concrete
provider (Brave/Tavily/SerpAPI/...) is adapted by pointing ``base_url`` at it and
mapping its fields to that contract.

The API key (if any) is resolved at call time from :mod:`common.secrets`, never
hardcoded (CLAUDE.md §7). Construction touches no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.http.client import HttpClient
from common.secrets import SecretProvider

__all__ = [
    "HostedSearchClient",
    "HostedSearchConfig",
    "HostedSearchResponse",
    "HostedSearchResult",
    "as_of_date_bound",
]


@dataclass(frozen=True)
class HostedSearchConfig:
    """Endpoint + field mapping for a hosted search provider."""

    base_url: str = "https://api.search.example/v1/search"
    provider: str = "hosted"
    version: str = "v1"
    api_key_secret: str | None = None
    api_key_header: str = "Authorization"
    query_param: str = "q"
    count_param: str = "count"
    offset_param: str = "offset"
    # Name of the provider's "results dated on or before" request param, if it
    # has one. When set, ``search(as_of=...)`` pushes the as-of ceiling into the
    # provider query itself (server-side bound, §2.1) instead of relying solely
    # on the client-side post-filter.
    as_of_param: str | None = None
    page_size: int = 10
    extra_params: dict[str, str] = field(default_factory=dict)


def as_of_date_bound(as_of: datetime) -> str:
    """Render an as-of ceiling as the provider-facing UTC date string.

    Day granularity: a provider date bound admits same-day-but-later results,
    so callers must still post-filter exact timestamps via ``filter_as_of``.
    """
    return as_of.astimezone(UTC).strftime("%Y-%m-%d")


class HostedSearchResult(BaseModel):
    """One provider result, before as-of filtering."""

    model_config = ConfigDict(frozen=True)

    title: str = ""
    url: str = ""
    content: str = ""
    published_date: str | None = None
    score: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_raw(cls, raw: Any) -> HostedSearchResult:
        """Tolerantly map a provider result object into the typed contract."""
        if not isinstance(raw, dict):
            return cls()
        published = raw.get("published_date")
        raw_score = raw.get("score")
        try:
            score = max(0.0, float(raw_score)) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        return cls(
            title=str(raw.get("title", "")),
            url=str(raw.get("url", "")),
            content=str(raw.get("content", "")),
            published_date=str(published) if published else None,
            score=score,
        )


class HostedSearchResponse(BaseModel):
    """Aggregated provider response with the raw pages retained for snapshotting."""

    model_config = ConfigDict(frozen=True)

    query: str
    results: tuple[HostedSearchResult, ...]
    raw: dict[str, Any]


class HostedSearchClient:
    """Calls a hosted search API and returns typed results.

    Pagination is offset-based and bounded by ``max_results``; the raw page
    payloads are retained on the response so the snapshot store can persist the
    provider's exact output for reproducibility (C3.3).
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        config: HostedSearchConfig | None = None,
        secrets: SecretProvider | None = None,
    ) -> None:
        self._http = http
        self._config = config or HostedSearchConfig()
        self._secrets = secrets

    @property
    def config(self) -> HostedSearchConfig:
        return self._config

    def _auth_headers(self) -> dict[str, str] | None:
        if self._config.api_key_secret and self._secrets is not None:
            key = self._secrets.get_secret(self._config.api_key_secret)
            return {self._config.api_key_header: key}
        return None

    def _fetch_page(
        self, query: str, *, count: int, offset: int, as_of: datetime | None = None
    ) -> tuple[list[Any], dict[str, Any]]:
        params: dict[str, Any] = {
            self._config.query_param: query,
            self._config.count_param: count,
            self._config.offset_param: offset,
            **self._config.extra_params,
        }
        if as_of is not None and self._config.as_of_param is not None:
            params[self._config.as_of_param] = as_of_date_bound(as_of)
        payload = self._http.get_json(
            self._config.base_url, params=params, headers=self._auth_headers()
        )
        raw = payload if isinstance(payload, dict) else {"results": []}
        results = raw.get("results", [])
        return (list(results) if isinstance(results, list) else []), raw

    def search(
        self, query: str, *, max_results: int = 10, as_of: datetime | None = None
    ) -> HostedSearchResponse:
        """Search for ``query``, returning up to ``max_results`` typed results.

        ``as_of`` pushes the knowledge-time ceiling into the provider request
        when the config maps a param for it (§2.1). It never *replaces* the
        client-side ``filter_as_of`` post-filter — the bound is day-granular
        and provider enforcement is not trusted on its own.
        """
        if max_results < 1:
            msg = "max_results must be >= 1."
            raise ValueError(msg)
        collected: list[HostedSearchResult] = []
        pages: list[dict[str, Any]] = []
        offset = 0
        while len(collected) < max_results:
            want = min(self._config.page_size, max_results - len(collected))
            raw_results, raw = self._fetch_page(query, count=want, offset=offset, as_of=as_of)
            pages.append(raw)
            if not raw_results:
                break
            collected.extend(HostedSearchResult.from_raw(r) for r in raw_results)
            offset += len(raw_results)
            if len(raw_results) < want:
                break
        return HostedSearchResponse(
            query=query,
            results=tuple(collected[:max_results]),
            raw={"pages": pages},
        )
