"""Secret resolution for DELPHI.

Credentials (API keys, DB passwords, sign-off tokens) must never live in code,
logs, or the repo (CLAUDE.md §7). This module provides a small, typed seam for
fetching secrets at runtime from either the process environment (local/dev) or
AWS Secrets Manager (production).

The :class:`SecretProvider` protocol keeps the rest of the codebase decoupled
from the backend. Tests inject an in-memory provider; production wires
:class:`AwsSecretsManagerProvider`.

Note: ``boto3`` is imported lazily via :func:`importlib.import_module` so it is
not a hard dependency and so static type checking does not require it to be
installed.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_ENV_PREFIX",
    "AwsSecretsManagerProvider",
    "EnvSecretProvider",
    "SecretNotFoundError",
    "SecretProvider",
]

DEFAULT_ENV_PREFIX = "DELPHI_SECRET_"


class SecretNotFoundError(KeyError):
    """Raised when a requested secret cannot be resolved."""


@runtime_checkable
class SecretProvider(Protocol):
    """Resolves a logical secret name to its plaintext value."""

    def get_secret(self, name: str) -> str: ...


def _env_key(prefix: str, name: str) -> str:
    """Map a logical secret name to its environment variable key."""
    return f"{prefix}{name.upper().replace('-', '_').replace('.', '_')}"


class EnvSecretProvider:
    """Resolve secrets from environment variables.

    A logical name like ``anthropic-api-key`` is read from the env var
    ``DELPHI_SECRET_ANTHROPIC_API_KEY`` (configurable via ``prefix``). Intended
    for local development and CI; production should prefer
    :class:`AwsSecretsManagerProvider`.
    """

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        prefix: str = DEFAULT_ENV_PREFIX,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._prefix = prefix

    def get_secret(self, name: str) -> str:
        key = _env_key(self._prefix, name)
        value = self._environ.get(key)
        if value is None:
            raise SecretNotFoundError(f"secret {name!r} not found (expected env var {key})")
        return value


class AwsSecretsManagerProvider:
    """Resolve secrets from AWS Secrets Manager.

    ``boto3`` is imported lazily, so importing this module never requires it.
    An explicit ``client`` may be injected for testing without AWS access.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                boto3 = importlib.import_module("boto3")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "boto3 is required for AwsSecretsManagerProvider; install it "
                    "or inject a client. Hint: `uv add boto3`."
                ) from exc
            self._client = boto3.client("secretsmanager", region_name=region_name)

    def get_secret(self, name: str) -> str:
        try:
            response = self._client.get_secret_value(SecretId=name)
        except Exception as exc:  # noqa: BLE001 - normalize backend errors
            raise SecretNotFoundError(f"secret {name!r} not retrievable: {exc}") from exc
        secret = response.get("SecretString")
        if secret is None:
            raise SecretNotFoundError(f"secret {name!r} has no SecretString payload")
        return secret
