"""Typed application settings for DELPHI.

Centralizes environment-derived configuration (Postgres DSN, AWS region, pinned
model identifiers, global budgets) behind one frozen, fully type-annotated
object. Modules should accept the values they need as arguments; this object is
the single sanctioned place that reads process environment for configuration.

Design contract (CLAUDE.md §7):
    - Config comes from typed settings, never magic constants scattered around.
    - Settings carry **no secrets**. Credentials are resolved separately via
      ``common.secrets`` (env or AWS Secrets Manager), never stored here.
    - ``Settings.from_env`` accepts an injected mapping so tests never mutate
      the global process environment (§8 determinism / no hidden global state).

Deliberately **not** included: any flag that could enable real-money trading.
Live-trading authorization is out-of-band only (``execution.live_stub`` reads
``DELPHI_LIVE_TRADING_AUTHORIZED`` directly) and must never be reachable through
a settings object an agent could populate (Prime Directive §2.1).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CLAUDE_FABLE_5_ID",
    "CLAUDE_OPUS_4_8_ID",
    "DEFAULT_GLOBAL_TRIALS_BUDGET",
    "DEFAULT_LLM_PROVIDER",
    "PG_DSN_ENV_VAR",
    "MissingSettingError",
    "Settings",
    "load_settings",
]

PG_DSN_ENV_VAR = "DELPHI_PG_DSN"
DEFAULT_GLOBAL_TRIALS_BUDGET = 100

# Pinned LLM identifiers, sourced from the Claude model cards (CLAUDE.md §7,
# never from memory) and verified against the docs in 2026-07. These are the
# **Claude API** model IDs (the default transport: the direct Anthropic API).
# Opus 4.8 is the general workhorse; Fable 5 is the strongest (autonomous,
# agentic) model, reserved for the meta-layer.
#
# If you re-enable AWS Bedrock (DELPHI_LLM_PROVIDER=bedrock), override the per-
# tier ids with Bedrock-style ids instead (e.g. "anthropic.claude-opus-4-8", or a
# cross-Region inference profile such as "us.anthropic.claude-fable-5").
CLAUDE_OPUS_4_8_ID = "claude-opus-4-8"
CLAUDE_FABLE_5_ID = "claude-fable-5"
DEFAULT_LLM_PROVIDER = "anthropic"


class MissingSettingError(RuntimeError):
    """Raised when a required setting is absent."""


def _opt(environ: Mapping[str, str], *names: str) -> str | None:
    """Return the first non-empty value among ``names`` in ``environ``."""
    for name in names:
        value = environ.get(name)
        if value:
            return value
    return None


def _as_bool(value: str | None) -> bool:
    """Interpret a truthy env string ('1'/'true'/'yes'/'on') as ``True``."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_EMBEDDING_DIM = 128


def _embedding_dim(environ: Mapping[str, str]) -> int:
    """Parse ``DELPHI_EMBEDDING_DIM`` as an int, defaulting to 128."""
    raw = _opt(environ, "DELPHI_EMBEDDING_DIM")
    if raw is None:
        return DEFAULT_EMBEDDING_DIM
    try:
        return int(raw)
    except ValueError as exc:
        raise MissingSettingError(f"DELPHI_EMBEDDING_DIM must be an integer, got {raw!r}") from exc


def _int_env(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = _opt(environ, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise MissingSettingError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = _opt(environ, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise MissingSettingError(f"{name} must be a number, got {raw!r}") from exc


class Settings(BaseModel):
    """Immutable, environment-derived application configuration.

    All fields are optional so that local/in-memory development and the test
    suite work with zero configuration. Call site helpers (e.g.
    :meth:`require_pg_dsn`) raise an explicit, actionable error when a value is
    actually needed but absent.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    pg_dsn: str | None = Field(
        default=None,
        description="PostgreSQL DSN for the DELPHI spine (registry, PIT, ledgers, caches).",
    )
    aws_region: str | None = Field(
        default=None,
        description="AWS region for Bedrock / Secrets Manager / S3.",
    )
    llm_provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        description="LLM transport provider: 'anthropic' (direct API, default) or 'bedrock'.",
    )

    # Model identifiers are pinned here (never hardcoded from memory — CLAUDE.md
    # §7) and mapped from the tier strings used by the agents. Tiered by
    # capability class: Opus 4.8 is the high-volume / mid workhorse ('opus'
    # tier), and Fable 5 (the strongest, most autonomous model) is reserved for
    # the strongest tier ('fable': supervisor / conductor / meta-layer). Env
    # vars override per tier: DELPHI_MODEL_OPUS / DELPHI_MODEL_FABLE.
    model_opus: str = Field(
        default=CLAUDE_OPUS_4_8_ID, description="Workhorse (fast/mid) tier model id."
    )
    model_fable: str = Field(default=CLAUDE_FABLE_5_ID, description="Strongest tier model id.")
    model_embedding: str | None = Field(default=None, description="Embedding model id.")
    embedding_dim: int = Field(
        default=128,
        ge=1,
        description=(
            "Embedding vector dimension; single source of truth for the pgvector "
            "memory column. 128 suits the deterministic floor; Titan V2 accepts "
            "256/512/1024. Fixed per deployment (changing it needs an index rebuild)."
        ),
    )

    global_trials_budget: int = Field(
        default=DEFAULT_GLOBAL_TRIALS_BUDGET,
        ge=0,
        description="Firm-wide multiple-testing trials budget enforced by orchestration.",
    )

    snapshot_dir: str | None = Field(
        default=None,
        description=(
            "Filesystem root for the durable evidence snapshot store (reproducible, "
            "leakage-auditable retrieval). None -> in-memory (non-persistent)."
        ),
    )

    calibration_artifact_path: str | None = Field(
        default=None,
        description=(
            "Path to the fitted calibration artifact JSON (written by `delphi "
            "calibration fit`, §2.5). None -> identity recalibrator + fixed alpha."
        ),
    )

    # Ensemble knobs (§3.4/§3.5). Production defaults: 4 method-agents x 3 runs
    # = 12 decorrelated draws, pooled in log-odds space, each draw reading a
    # seeded ~80% evidence subset.
    runs_per_agent: int = Field(
        default=3,
        ge=1,
        description="Ensemble draws per method-agent (total draws = 4 agents x this).",
    )
    aggregator: str = Field(
        default="log_odds_trimmed_mean",
        description=(
            "Ensemble aggregator: log_odds_trimmed_mean | log_odds_mean | median | trimmed_mean."
        ),
    )
    evidence_subset_fraction: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Per-draw seeded evidence subsample fraction (1.0 disables).",
    )

    # Agentic search knobs (§1: search quality dominates). Rounds > 1 enables
    # the LLM-directed iterative retrieval loop around the base searcher.
    search_rounds: int = Field(
        default=3,
        ge=1,
        description="Agentic search rounds per question (1 = single-shot, no planner).",
    )
    search_queries: int = Field(
        default=8,
        ge=1,
        description="Hard cap on provider queries per agentic search invocation.",
    )
    subquestion_searches: int = Field(
        default=3,
        ge=0,
        description="Decomposition sub-questions given their own search pass (0 disables).",
    )

    # Async forecast-job surface (api/jobs.py): hosted front-ends cap request
    # time (App Runner: 120s hard), so forecasts run out-of-request on a small
    # per-process worker pool and clients poll for the result.
    job_workers: int = Field(
        default=2,
        ge=1,
        description="Concurrent async forecast jobs per API process.",
    )
    job_stale_after_s: int = Field(
        default=1800,
        ge=1,
        description=(
            "Seconds a running forecast job may go without completing before "
            "polls report it failed (worker lost to a crash/restart)."
        ),
    )

    # Data-ingestion HTTP identity. SEC EDGAR mandates a descriptive User-Agent
    # (e.g. "DELPHI Research research@example.com"); other sources accept any.
    http_user_agent: str | None = Field(
        default=None,
        description="Default HTTP User-Agent for source ingestion adapters.",
    )
    edgar_user_agent: str | None = Field(
        default=None,
        description="Descriptive User-Agent required by SEC EDGAR (overrides http_user_agent).",
    )

    # Deployment observability + artifact wiring (Phase 4). All optional so the
    # test suite and local runs need zero configuration. None of these carry
    # secrets; credentials are resolved separately via ``common.secrets``.
    sns_alert_topic_arn: str | None = Field(
        default=None,
        description="SNS topic ARN for routing monitoring/orchestration alerts.",
    )
    s3_data_lake_bucket: str | None = Field(
        default=None,
        description="S3 bucket name for the raw/historical PIT data lake.",
    )
    s3_artifacts_bucket: str | None = Field(
        default=None,
        description="S3 bucket name for experiment/backtest artifacts.",
    )
    log_json: bool = Field(
        default=False,
        description="Render logs as JSON (production/CloudWatch) vs console (local).",
    )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from an environment mapping.

        Args:
            environ: Mapping to read from. Defaults to ``os.environ``. Tests
                should pass an explicit dict to avoid mutating global state.
        """
        env = os.environ if environ is None else environ
        budget_raw = _opt(env, "DELPHI_GLOBAL_TRIALS_BUDGET")
        if budget_raw is None:
            budget = DEFAULT_GLOBAL_TRIALS_BUDGET
        else:
            try:
                budget = int(budget_raw)
            except ValueError as exc:
                raise MissingSettingError(
                    f"DELPHI_GLOBAL_TRIALS_BUDGET must be an integer, got {budget_raw!r}"
                ) from exc
        return cls(
            pg_dsn=_opt(env, PG_DSN_ENV_VAR),
            aws_region=_opt(env, "DELPHI_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
            llm_provider=_opt(env, "DELPHI_LLM_PROVIDER") or DEFAULT_LLM_PROVIDER,
            model_opus=_opt(env, "DELPHI_MODEL_OPUS") or CLAUDE_OPUS_4_8_ID,
            model_fable=_opt(env, "DELPHI_MODEL_FABLE") or CLAUDE_FABLE_5_ID,
            model_embedding=_opt(env, "DELPHI_MODEL_EMBEDDING"),
            embedding_dim=_embedding_dim(env),
            global_trials_budget=budget,
            snapshot_dir=_opt(env, "DELPHI_SNAPSHOT_DIR"),
            calibration_artifact_path=_opt(env, "DELPHI_CALIBRATION_ARTIFACT"),
            runs_per_agent=_int_env(env, "DELPHI_RUNS_PER_AGENT", 3),
            aggregator=_opt(env, "DELPHI_AGGREGATOR") or "log_odds_trimmed_mean",
            evidence_subset_fraction=_float_env(env, "DELPHI_EVIDENCE_SUBSET_FRACTION", 0.8),
            search_rounds=_int_env(env, "DELPHI_SEARCH_ROUNDS", 3),
            search_queries=_int_env(env, "DELPHI_SEARCH_QUERIES", 8),
            subquestion_searches=_int_env(env, "DELPHI_SUBQUESTION_SEARCHES", 3),
            job_workers=_int_env(env, "DELPHI_JOB_WORKERS", 2),
            job_stale_after_s=_int_env(env, "DELPHI_JOB_TIMEOUT_S", 1800),
            http_user_agent=_opt(env, "DELPHI_HTTP_USER_AGENT"),
            edgar_user_agent=_opt(env, "DELPHI_EDGAR_USER_AGENT"),
            sns_alert_topic_arn=_opt(env, "DELPHI_SNS_ALERT_TOPIC_ARN"),
            s3_data_lake_bucket=_opt(env, "DELPHI_S3_DATA_LAKE_BUCKET"),
            s3_artifacts_bucket=_opt(env, "DELPHI_S3_ARTIFACTS_BUCKET"),
            log_json=_as_bool(_opt(env, "DELPHI_LOG_JSON")),
        )

    def model_for_tier(self, tier: str) -> str:
        """Resolve a capability tier ('opus'/'fable') to its model id.

        'opus' is the workhorse tier (Opus 4.8 by default; high-volume
        estimation and mid reasoning); 'fable' is the strongest tier (Fable 5
        by default; supervisor / conductor / meta-layer). Kept string-keyed so
        callers need not share an enum type for the tier names. Raises
        ``KeyError`` on an unknown tier.
        """
        return {
            "opus": self.model_opus,
            "fable": self.model_fable,
        }[tier]

    def require_pg_dsn(self) -> str:
        """Return the Postgres DSN or raise if it is not configured."""
        if not self.pg_dsn:
            raise MissingSettingError(
                f"{PG_DSN_ENV_VAR} is not set. Export a PostgreSQL DSN, e.g. "
                f'export {PG_DSN_ENV_VAR}="postgresql://postgres:delphi@localhost:5432/delphi"'
            )
        return self.pg_dsn


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Convenience wrapper around :meth:`Settings.from_env`."""
    return Settings.from_env(environ)
