"""ForecastBench dataset fetcher (C7.2 network half).

Pulls ForecastBench question/resolution JSON (the public dataset repo; no auth)
and maps it into the plain-``dict`` records
:class:`benchmarks.forecastbench.ForecastBenchAdapter.from_records` expects. As
with the Metaculus fetcher, the network read is kept separate from the
deterministic mapping (:func:`map_question`) so mapping is unit-tested without the
network and the fetcher is exercised hermetically via ``httpx.MockTransport``.

As-of discipline (Prime Directive §2.1): each record's ``as_of`` is the question's
``freeze_datetime`` (or the set's ``forecast_due_date``) — the point at which the
crowd/market value was frozen and the answer was unknown — never ``now()``. The
freeze value is carried as ``freeze_value`` so a crowd-consensus baseline can be
built downstream; ForecastBench also supplies human baselines the suite loader can
surface alongside the model score.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from common.http.client import HttpClient

__all__ = [
    "ForecastBenchFetcher",
    "map_question",
    "map_resolutions",
]

_DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"
)


_MISSING_SENTINELS = {"", "n/a", "na", "none", "null", "-"}


def _first(*values: Any) -> Any:
    """Return the first non-empty value (skips ``None`` and empty strings)."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _clean_dt(value: Any) -> Any:
    """Normalize a date-ish field: map ForecastBench sentinels (e.g. ``N/A``) to None.

    Market questions carry ``N/A`` for absent close/resolution dates; passing that
    to the adapter's ISO parser would raise, so it is treated as missing here.
    """
    if isinstance(value, str) and value.strip().lower() in _MISSING_SENTINELS:
        return None
    return value


def _resolution_key(source: str, raw_id: Any) -> str:
    """Composite external id used to join questions to resolutions."""
    source = source.strip()
    return f"{source}-{raw_id}" if source else str(raw_id)


def map_resolutions(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a resolution document by composite external id.

    Entries explicitly marked ``resolved: false`` (or lacking a value/date) are
    skipped so only genuinely-resolved questions produce a resolution row.
    """
    entries = doc.get("resolutions")
    if not isinstance(entries, Sequence):
        return {}
    resolved: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("resolved") is False:
            continue
        raw_id = _first(entry.get("id"), entry.get("question_id"))
        value = _first(entry.get("resolved_to"), entry.get("resolved_value"), entry.get("value"))
        resolved_at = _clean_dt(_first(entry.get("resolution_date"), entry.get("resolved_at")))
        if raw_id is None or value is None or resolved_at is None:
            continue
        source = str(entry.get("source", ""))
        try:
            resolved_value = float(value)
        except (TypeError, ValueError):
            continue
        resolved[_resolution_key(source, raw_id)] = {
            "resolved_value": resolved_value,
            "resolved_at": resolved_at,
            "resolution_source": _first(source, "forecastbench"),
        }
    return resolved


def map_question(
    question: Mapping[str, Any],
    *,
    default_as_of: Any = None,
    resolutions: Mapping[str, Mapping[str, Any]] | None = None,
    freeze_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Map one ForecastBench question into an adapter record, or ``None`` to skip."""
    raw_id = _first(question.get("id"), question.get("question_id"))
    text = _first(question.get("question"), question.get("title"))
    if raw_id is None or text is None:
        return None

    source = str(question.get("source", ""))
    external_id = _resolution_key(source, raw_id)

    if freeze_at is not None:
        as_of: Any = freeze_at.isoformat()
    else:
        as_of = _first(_clean_dt(question.get("freeze_datetime")), _clean_dt(default_as_of))
    if as_of is None:
        return None

    record: dict[str, Any] = {
        "id": external_id,
        "question": text,
        "as_of": as_of,
        "domain": str(_first(question.get("category"), source, "general")),
        "resolution_criteria": str(_first(question.get("resolution_criteria"), "") or ""),
    }
    close_time = _first(
        _clean_dt(question.get("resolution_date")),
        _clean_dt(question.get("market_info_close_datetime")),
    )
    if close_time is not None:
        record["close_time"] = close_time

    freeze_value = question.get("freeze_datetime_value")
    if freeze_value is not None:
        with contextlib.suppress(TypeError, ValueError):
            record["freeze_value"] = float(freeze_value)

    resolution = (resolutions or {}).get(external_id)
    if resolution is not None:
        record["resolved_value"] = resolution["resolved_value"]
        record["resolved_at"] = resolution["resolved_at"]
        record["resolution_source"] = resolution.get("resolution_source", "forecastbench")
    return record


class ForecastBenchFetcher:
    """Fetches ForecastBench question/resolution JSON and maps it to adapter records.

    Network access goes through the injected :class:`HttpClient`. The dataset repo
    is public, so no credentials are required. ``question_set`` / ``resolution_set``
    may be repo-relative paths (joined onto ``base_url``) or absolute URLs.
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    def _url(self, ref: str) -> str:
        if ref.startswith(("http://", "https://")):
            return ref
        return f"{self._base_url}/{ref.lstrip('/')}"

    def fetch(
        self,
        *,
        question_set: str,
        resolution_set: str | None = None,
        freeze_at: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch a question set (and optional resolution set) as adapter records."""
        qdoc = self._http.get_json(self._url(question_set))
        qdoc = qdoc if isinstance(qdoc, Mapping) else {}
        questions = qdoc.get("questions", [])
        default_as_of = _first(qdoc.get("forecast_due_date"), qdoc.get("freeze_datetime"))

        resolutions: dict[str, dict[str, Any]] = {}
        if resolution_set is not None:
            rdoc = self._http.get_json(self._url(resolution_set))
            resolutions = map_resolutions(rdoc if isinstance(rdoc, Mapping) else {})

        records: list[dict[str, Any]] = []
        if isinstance(questions, Sequence):
            for question in questions:
                if isinstance(question, Mapping):
                    record = map_question(
                        question,
                        default_as_of=default_as_of,
                        resolutions=resolutions,
                        freeze_at=freeze_at,
                    )
                    if record is not None:
                        records.append(record)
        return records
