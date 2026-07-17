"""Generic HTTP transport layer (domain-agnostic core, CLAUDE.md section 11).

Kept out of ``common/__init__`` so importing ``common`` never pulls httpx.
Import directly: ``from common.http import HttpClient``.
"""

from __future__ import annotations

from common.http.client import HttpClient
from common.http.config import HttpConfig
from common.http.errors import (
    HttpError,
    HttpNotFound,
    HttpRateLimited,
    HttpTransportFailure,
)

__all__ = [
    "HttpClient",
    "HttpConfig",
    "HttpError",
    "HttpNotFound",
    "HttpRateLimited",
    "HttpTransportFailure",
]
