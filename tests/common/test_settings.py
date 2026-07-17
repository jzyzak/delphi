"""Unit tests for common.settings (§8: happy path, edges, failure modes)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.settings import (
    CLAUDE_FABLE_5_ID,
    CLAUDE_OPUS_4_8_ID,
    DEFAULT_GLOBAL_TRIALS_BUDGET,
    DEFAULT_LLM_PROVIDER,
    PG_DSN_ENV_VAR,
    MissingSettingError,
    Settings,
    load_settings,
)


def test_from_env_reads_all_fields() -> None:
    env = {
        PG_DSN_ENV_VAR: "postgresql://u:p@host:5432/delphi",
        "DELPHI_AWS_REGION": "us-east-1",
        "DELPHI_LLM_PROVIDER": "anthropic",
        "DELPHI_MODEL_OPUS": "opus-id",
        "DELPHI_MODEL_FABLE": "fable-id",
        "DELPHI_MODEL_EMBEDDING": "embed-id",
        "DELPHI_GLOBAL_TRIALS_BUDGET": "250",
    }
    settings = Settings.from_env(env)
    assert settings.pg_dsn == "postgresql://u:p@host:5432/delphi"
    assert settings.aws_region == "us-east-1"
    assert settings.llm_provider == "anthropic"
    assert settings.model_opus == "opus-id"
    assert settings.model_fable == "fable-id"
    assert settings.model_embedding == "embed-id"
    assert settings.global_trials_budget == 250


def test_from_env_defaults_when_empty() -> None:
    settings = Settings.from_env({})
    assert settings.pg_dsn is None
    assert settings.aws_region is None
    assert settings.global_trials_budget == DEFAULT_GLOBAL_TRIALS_BUDGET
    assert settings.snapshot_dir is None


def test_snapshot_dir_read_from_env() -> None:
    settings = Settings.from_env({"DELPHI_SNAPSHOT_DIR": "/var/delphi/snap"})
    assert settings.snapshot_dir == "/var/delphi/snap"


def test_calibration_artifact_path_defaults_to_none() -> None:
    assert Settings.from_env({}).calibration_artifact_path is None


def test_calibration_artifact_path_read_from_env() -> None:
    settings = Settings.from_env({"DELPHI_CALIBRATION_ARTIFACT": "/var/delphi/cal.json"})
    assert settings.calibration_artifact_path == "/var/delphi/cal.json"


def test_ensemble_knob_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.runs_per_agent == 3
    assert settings.aggregator == "log_odds_trimmed_mean"
    assert settings.evidence_subset_fraction == 0.8


def test_ensemble_knobs_read_from_env() -> None:
    settings = Settings.from_env(
        {
            "DELPHI_RUNS_PER_AGENT": "5",
            "DELPHI_AGGREGATOR": "median",
            "DELPHI_EVIDENCE_SUBSET_FRACTION": "1.0",
        }
    )
    assert settings.runs_per_agent == 5
    assert settings.aggregator == "median"
    assert settings.evidence_subset_fraction == 1.0


def test_search_knob_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.search_rounds == 3
    assert settings.search_queries == 8
    assert settings.subquestion_searches == 3


def test_search_knobs_read_from_env() -> None:
    settings = Settings.from_env(
        {
            "DELPHI_SEARCH_ROUNDS": "1",
            "DELPHI_SEARCH_QUERIES": "20",
            "DELPHI_SUBQUESTION_SEARCHES": "0",
        }
    )
    assert settings.search_rounds == 1
    assert settings.search_queries == 20
    assert settings.subquestion_searches == 0


def test_search_rounds_must_be_integer() -> None:
    with pytest.raises(MissingSettingError, match="DELPHI_SEARCH_ROUNDS"):
        Settings.from_env({"DELPHI_SEARCH_ROUNDS": "many"})


def test_job_knob_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.job_workers == 2
    assert settings.job_stale_after_s == 1800


def test_job_knobs_read_from_env() -> None:
    settings = Settings.from_env(
        {
            "DELPHI_JOB_WORKERS": "4",
            "DELPHI_JOB_TIMEOUT_S": "3600",
        }
    )
    assert settings.job_workers == 4
    assert settings.job_stale_after_s == 3600


def test_job_workers_must_be_integer() -> None:
    with pytest.raises(MissingSettingError, match="DELPHI_JOB_WORKERS"):
        Settings.from_env({"DELPHI_JOB_WORKERS": "a few"})


def test_job_timeout_must_be_integer() -> None:
    with pytest.raises(MissingSettingError, match="DELPHI_JOB_TIMEOUT_S"):
        Settings.from_env({"DELPHI_JOB_TIMEOUT_S": "an hour"})


def test_runs_per_agent_must_be_integer() -> None:
    with pytest.raises(MissingSettingError, match="DELPHI_RUNS_PER_AGENT"):
        Settings.from_env({"DELPHI_RUNS_PER_AGENT": "three"})


def test_evidence_subset_fraction_must_be_number() -> None:
    with pytest.raises(MissingSettingError, match="DELPHI_EVIDENCE_SUBSET_FRACTION"):
        Settings.from_env({"DELPHI_EVIDENCE_SUBSET_FRACTION": "most"})


def test_empty_env_pins_each_tier_to_its_capability_class() -> None:
    settings = Settings.from_env({})
    # Default transport is the direct Anthropic (Claude) API.
    assert settings.llm_provider == DEFAULT_LLM_PROVIDER == "anthropic"
    assert settings.model_opus == CLAUDE_OPUS_4_8_ID == "claude-opus-4-8"
    assert settings.model_fable == CLAUDE_FABLE_5_ID == "claude-fable-5"
    assert settings.model_embedding is None


def test_env_overrides_pinned_model_ids() -> None:
    env = {
        "DELPHI_LLM_PROVIDER": "anthropic",
        "DELPHI_MODEL_OPUS": "custom-opus",
    }
    settings = Settings.from_env(env)
    assert settings.llm_provider == "anthropic"
    assert settings.model_opus == "custom-opus"
    # Unset tier still falls back to its pinned capability-class id.
    assert settings.model_fable == CLAUDE_FABLE_5_ID


def test_empty_string_model_falls_back_to_pinned_default() -> None:
    settings = Settings.from_env({"DELPHI_MODEL_FABLE": ""})
    assert settings.model_fable == CLAUDE_FABLE_5_ID


def test_model_for_tier_resolves_each_tier() -> None:
    settings = Settings.from_env(
        {
            "DELPHI_MODEL_OPUS": "o-id",
            "DELPHI_MODEL_FABLE": "f-id",
        }
    )
    assert settings.model_for_tier("opus") == "o-id"
    assert settings.model_for_tier("fable") == "f-id"


def test_model_for_tier_unknown_raises() -> None:
    with pytest.raises(KeyError):
        Settings.from_env({}).model_for_tier("gpt")
    # Retired legacy tier names must not silently resolve.
    with pytest.raises(KeyError):
        Settings.from_env({}).model_for_tier("haiku")
    with pytest.raises(KeyError):
        Settings.from_env({}).model_for_tier("sonnet")


def test_aws_region_fallback_order() -> None:
    # DELPHI_AWS_REGION wins over AWS_REGION which wins over AWS_DEFAULT_REGION.
    assert Settings.from_env({"AWS_DEFAULT_REGION": "eu-west-1"}).aws_region == "eu-west-1"
    assert (
        Settings.from_env({"AWS_DEFAULT_REGION": "eu-west-1", "AWS_REGION": "us-east-2"}).aws_region
        == "us-east-2"
    )
    assert (
        Settings.from_env({"AWS_REGION": "us-east-2", "DELPHI_AWS_REGION": "us-west-1"}).aws_region
        == "us-west-1"
    )


def test_empty_string_treated_as_unset() -> None:
    assert Settings.from_env({PG_DSN_ENV_VAR: ""}).pg_dsn is None


def test_require_pg_dsn_returns_value() -> None:
    settings = Settings.from_env({PG_DSN_ENV_VAR: "postgresql://localhost/delphi"})
    assert settings.require_pg_dsn() == "postgresql://localhost/delphi"


def test_require_pg_dsn_raises_when_missing() -> None:
    with pytest.raises(MissingSettingError, match=PG_DSN_ENV_VAR):
        Settings.from_env({}).require_pg_dsn()


def test_invalid_budget_raises() -> None:
    with pytest.raises(MissingSettingError, match="must be an integer"):
        Settings.from_env({"DELPHI_GLOBAL_TRIALS_BUDGET": "lots"})


def test_settings_is_frozen() -> None:
    settings = Settings.from_env({})
    with pytest.raises(ValidationError):
        settings.pg_dsn = "mutated"  # type: ignore[misc]


def test_load_settings_wrapper() -> None:
    assert load_settings({PG_DSN_ENV_VAR: "postgresql://x/y"}).pg_dsn == "postgresql://x/y"
