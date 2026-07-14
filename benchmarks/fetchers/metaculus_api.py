"""Metaculus API fetcher (C7.3 network half).

Pulls posts from the Metaculus API and maps them into the plain-``dict`` records
:class:`benchmarks.metaculus.MetaculusAdapter.from_records` expects. The network
call (paging ``/posts/``) is kept separate from the deterministic mapping
(:func:`map_post`) so the mapping is unit-tested without the network and the whole
fetcher is exercised hermetically via an injected ``httpx.MockTransport`` client.

As-of discipline (Prime Directive §2.1): every emitted record carries an explicit
``as_of``. By default that is the question's ``open_time`` — a moment when the
answer was genuinely unknown — never ``now()``. A ``freeze_at`` override pins all
questions to one snapshot instant instead. The community prediction is retained so
a crowd-consensus baseline can be derived downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from common.http.client import HttpClient
from common.secrets import SecretNotFoundError, SecretProvider

__all__ = [
    "METACULUS_API_TOKEN_SECRET",
    "MetaculusFetcher",
    "map_post",
]

METACULUS_API_TOKEN_SECRET = "metaculus-api-token"
_DEFAULT_BASE_URL = "https://www.metaculus.com/api"
# Metaculus binary questions resolve to one of these labels.
_BINARY_RESOLUTION = {"yes": 1.0, "no": 0.0}


def _first(*values: Any) -> Any:
    """Return the first non-empty value (skips ``None`` and empty strings)."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _community_prediction(question: Mapping[str, Any]) -> float | None:
    """Extract the recency-weighted community probability for a binary question."""
    aggregations = question.get("aggregations")
    if not isinstance(aggregations, Mapping):
        return None
    recency = aggregations.get("recency_weighted")
    if not isinstance(recency, Mapping):
        return None
    latest = recency.get("latest")
    if not isinstance(latest, Mapping):
        return None
    centers = latest.get("centers")
    if isinstance(centers, Sequence) and centers:
        try:
            return float(centers[0])
        except (TypeError, ValueError):
            return None
    return None


def _domain(post: Mapping[str, Any]) -> str:
    """Derive a coarse domain label from the post's projects/categories."""
    projects = post.get("projects")
    if isinstance(projects, Mapping):
        categories = projects.get("category")
        if isinstance(categories, Sequence) and categories:
            first = categories[0]
            if isinstance(first, Mapping):
                slug = _first(first.get("slug"), first.get("name"))
                if slug is not None:
                    return str(slug)
    return "general"


def map_post(
    post: Mapping[str, Any],
    *,
    binary_only: bool = True,
    freeze_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Map one Metaculus post into an adapter record, or ``None`` to skip it.

    A record is skipped when it lacks an id/title, has no derivable ``as_of``, or
    (when ``binary_only``) is not a binary question.
    """
    question = post.get("question")
    question = question if isinstance(question, Mapping) else {}

    question_type = str(_first(question.get("type"), post.get("type"), "binary"))
    if binary_only and question_type != "binary":
        return None

    external_id = _first(post.get("id"), question.get("id"))
    title = _first(post.get("title"), question.get("title"))
    if external_id is None or title is None:
        return None

    if freeze_at is not None:
        as_of: Any = freeze_at.isoformat()
    else:
        as_of = _first(
            question.get("open_time"),
            post.get("open_time"),
            post.get("published_at"),
        )
    if as_of is None:
        return None  # cannot pin an as-of safely -> drop rather than guess.

    record: dict[str, Any] = {
        "id": external_id,
        "title": title,
        "as_of": as_of,
        "question_type": question_type,
        "domain": _domain(post),
        "resolution_criteria": str(_first(question.get("description"), "") or ""),
    }
    close_time = _first(question.get("scheduled_close_time"), post.get("scheduled_close_time"))
    if close_time is not None:
        record["close_time"] = close_time

    community = _community_prediction(question)
    if community is not None:
        record["community"] = community

    resolution = question.get("resolution")
    resolved_value = _BINARY_RESOLUTION.get(str(resolution).lower()) if resolution else None
    resolved_at = _first(question.get("actual_resolve_time"), question.get("actual_close_time"))
    if resolved_value is not None and resolved_at is not None:
        record["resolution"] = resolved_value
        record["resolved_at"] = resolved_at

    return record


class MetaculusFetcher:
    """Fetches Metaculus posts and maps them to adapter records.

    Network access goes through the injected :class:`HttpClient`; an optional
    :class:`SecretProvider` supplies the API token (logical name
    ``metaculus-api-token``) as ``Authorization: Token <key>``. Public reads work
    without a token, so a missing secret is tolerated rather than fatal.
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        base_url: str = _DEFAULT_BASE_URL,
        secrets: SecretProvider | None = None,
        api_token_secret: str = METACULUS_API_TOKEN_SECRET,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._secrets = secrets
        self._api_token_secret = api_token_secret

    def _auth_headers(self) -> dict[str, str] | None:
        if self._secrets is None:
            return None
        try:
            token = self._secrets.get_secret(self._api_token_secret)
        except SecretNotFoundError:
            return None
        return {"Authorization": f"Token {token}"}

    def fetch(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        max_pages: int = 1,
        binary_only: bool = True,
        freeze_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Page the posts endpoint and return mapped adapter records.

        ``params`` are forwarded to the first request (e.g. status/order/limit
        filters). Paging follows the API's ``next`` link up to ``max_pages``.
        """
        if max_pages < 1:
            msg = "max_pages must be >= 1."
            raise ValueError(msg)
        headers = self._auth_headers()
        url: str | None = f"{self._base_url}/posts/"
        query: Mapping[str, Any] | None = params
        records: list[dict[str, Any]] = []
        pages = 0
        while url is not None and pages < max_pages:
            payload = self._http.get_json(url, params=query, headers=headers)
            page = payload if isinstance(payload, Mapping) else {}
            results = page.get("results", [])
            if isinstance(results, Sequence):
                for post in results:
                    if isinstance(post, Mapping):
                        record = map_post(post, binary_only=binary_only, freeze_at=freeze_at)
                        if record is not None:
                            records.append(record)
            nxt = page.get("next")
            url = str(nxt) if isinstance(nxt, str) and nxt else None
            query = None  # the `next` link already carries the query string.
            pages += 1
        return records
