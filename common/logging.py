"""Structured logging configuration for DELPHI services.

Library code logs through ``structlog.get_logger`` (CLAUDE.md §7 — no bare
``print``). This module configures the rendering *once*, at a service entry
point, so:

* in production (ECS/Batch) logs render as one JSON object per line, which
  CloudWatch Logs ingests and indexes natively;
* locally, logs render as colourised console lines for readability.

The configuration is intentionally global and idempotent — it is a process-level
concern owned by the entry point, never by library modules or tests.
"""

from __future__ import annotations

import logging

import structlog

__all__ = ["configure_logging"]

_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def configure_logging(*, json_logs: bool = True, level: str = "info") -> None:
    """Configure ``structlog`` process-wide rendering.

    Args:
        json_logs: emit one JSON object per line (production/CloudWatch) when
            ``True``; a human-readable console renderer otherwise.
        level: minimum level name ('debug'..'critical'). Unknown names fall back
            to 'info' rather than raising, so a misconfigured env var never
            crashes a service on startup.
    """
    min_level = _LEVELS.get(level.lower(), logging.INFO)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
