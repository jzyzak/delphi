"""Gating fixtures for the live smoke suite.

These tests hit real infrastructure (the Anthropic API, Tavily, Postgres) and
cost money + latency, so they are OFF by default. They run only when
``DELPHI_LIVE_SMOKE=1`` is set, and each per-dependency fixture skips
individually when its own env is absent. Nothing here runs in CI or the default
``pytest`` invocation, so the coverage gate is unaffected. This is the operator's
"does everything actually work?" suite, not a development test.

Deliberate exception to the hermeticity rule (tests/conftest.py): this suite
probes the REAL production dependencies read-only — it never truncates or
writes fixture rows — so it uses ``DELPHI_PG_DSN`` via settings on purpose.
"""

from __future__ import annotations

import os

import pytest

from common.settings import Settings, load_settings

LIVE_ENABLED = os.environ.get("DELPHI_LIVE_SMOKE") == "1"


@pytest.fixture(autouse=True)
def _require_live_enabled() -> None:
    """Skip every live test unless the operator opted in explicitly."""
    if not LIVE_ENABLED:
        pytest.skip("live smoke suite disabled; set DELPHI_LIVE_SMOKE=1 to run")


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def require_postgres(settings: Settings) -> str:
    if not settings.pg_dsn:
        pytest.skip("DELPHI_PG_DSN not set")
    return settings.pg_dsn


@pytest.fixture
def require_llm(settings: Settings) -> Settings:
    # Default transport is the direct Anthropic API; it needs the Claude key.
    # (If DELPHI_LLM_PROVIDER=bedrock, this instead needs DELPHI_AWS_REGION.)
    if settings.llm_provider == "bedrock":
        if not settings.aws_region:
            pytest.skip("DELPHI_AWS_REGION not set (bedrock provider)")
    elif not os.environ.get("DELPHI_SECRET_ANTHROPIC_API_KEY"):
        pytest.skip("DELPHI_SECRET_ANTHROPIC_API_KEY not set")
    return settings


@pytest.fixture
def require_tavily() -> None:
    if not os.environ.get("DELPHI_SECRET_TAVILY_API_KEY"):
        pytest.skip("DELPHI_SECRET_TAVILY_API_KEY not set")
