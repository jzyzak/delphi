"""Unit tests for common.secrets (§8: happy path, edges, failure modes)."""

from __future__ import annotations

import pytest

from common.secrets import (
    AwsSecretsManagerProvider,
    EnvSecretProvider,
    SecretNotFoundError,
    SecretProvider,
)


class TestEnvSecretProvider:
    def test_resolves_with_default_prefix(self) -> None:
        provider = EnvSecretProvider({"DELPHI_SECRET_ANTHROPIC_API_KEY": "sk-123"})
        assert provider.get_secret("anthropic-api-key") == "sk-123"

    def test_name_normalization(self) -> None:
        provider = EnvSecretProvider({"DELPHI_SECRET_DB_PASSWORD": "pw"})
        # dots and dashes normalize to underscores, case-insensitive
        assert provider.get_secret("db.password") == "pw"
        assert provider.get_secret("DB-PASSWORD") == "pw"

    def test_custom_prefix(self) -> None:
        provider = EnvSecretProvider({"MY_TOKEN": "abc"}, prefix="MY_")
        assert provider.get_secret("token") == "abc"

    def test_missing_secret_raises(self) -> None:
        provider = EnvSecretProvider({})
        with pytest.raises(SecretNotFoundError, match="DELPHI_SECRET_MISSING"):
            provider.get_secret("missing")

    def test_satisfies_protocol(self) -> None:
        provider: SecretProvider = EnvSecretProvider({})
        assert isinstance(provider, SecretProvider)


class _FakeSecretsClient:
    def __init__(self, payload: dict[str, dict[str, str]]) -> None:
        self._payload = payload

    def get_secret_value(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803 - AWS API name
        if SecretId not in self._payload:
            raise RuntimeError(f"no such secret {SecretId}")
        return self._payload[SecretId]


class TestAwsSecretsManagerProvider:
    def test_resolves_secret_string(self) -> None:
        client = _FakeSecretsClient({"prod/anthropic": {"SecretString": "sk-live"}})
        provider = AwsSecretsManagerProvider(client=client)
        assert provider.get_secret("prod/anthropic") == "sk-live"

    def test_missing_secret_normalized(self) -> None:
        provider = AwsSecretsManagerProvider(client=_FakeSecretsClient({}))
        with pytest.raises(SecretNotFoundError, match="not retrievable"):
            provider.get_secret("absent")

    def test_no_secret_string_payload(self) -> None:
        client = _FakeSecretsClient({"binary/only": {"SecretBinary": "..."}})
        provider = AwsSecretsManagerProvider(client=client)
        with pytest.raises(SecretNotFoundError, match="no SecretString"):
            provider.get_secret("binary/only")

    def test_missing_boto3_raises_helpful_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "boto3":
                raise ModuleNotFoundError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib, "import_module", fake_import)
        with pytest.raises(RuntimeError, match="boto3 is required"):
            AwsSecretsManagerProvider(region_name="us-east-1")
