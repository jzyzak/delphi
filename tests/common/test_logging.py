"""Unit tests for structured logging configuration."""

from __future__ import annotations

import structlog

from common.logging import configure_logging


def test_configure_logging_json_is_configured() -> None:
    structlog.reset_defaults()
    configure_logging(json_logs=True, level="info")
    assert structlog.is_configured()
    # A logger call must not raise under the JSON renderer.
    structlog.get_logger("test").info("hello", k=1)


def test_configure_logging_console_mode() -> None:
    structlog.reset_defaults()
    configure_logging(json_logs=False, level="debug")
    assert structlog.is_configured()


def test_configure_logging_unknown_level_falls_back() -> None:
    structlog.reset_defaults()
    # An unknown level name must not raise; it falls back to info.
    configure_logging(json_logs=True, level="not-a-level")
    assert structlog.is_configured()
    structlog.reset_defaults()
