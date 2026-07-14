"""Tests for compliance features (C10.4)."""

from __future__ import annotations

import pytest

from api.compliance import (
    ComplianceOptions,
    ProviderOptOutError,
    filter_providers,
    should_retain,
    usage_for,
)


def test_filter_providers_keeps_permitted() -> None:
    opts = ComplianceOptions(provider_opt_out=frozenset({"openai"}))
    assert filter_providers(("bedrock", "openai"), opts) == ("bedrock",)


def test_filter_providers_all_opted_out_fails_closed() -> None:
    opts = ComplianceOptions(provider_opt_out=frozenset({"bedrock"}))
    with pytest.raises(ProviderOptOutError):
        filter_providers(("bedrock",), opts)


def test_should_retain() -> None:
    assert should_retain(ComplianceOptions())
    assert not should_retain(ComplianceOptions(retention_opt_out=True))


def test_usage_cost_none_without_price() -> None:
    usage = usage_for("delphi", model_calls=3)
    assert usage.cost_usd is None
    assert usage.model_calls == 3


def test_usage_cost_computed_with_price() -> None:
    usage = usage_for("delphi_deep", model_calls=4, price_per_call=0.01)
    assert usage.cost_usd == pytest.approx(0.04)
