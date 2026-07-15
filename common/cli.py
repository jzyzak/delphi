"""``delphi`` command-line entry point.

The full §8 CLI surface: ``intake``, ``forecast``, ``resolve``, ``eval``,
``conductor``, ``bench live``, ``serve``, and ``doctor``. Every dependency (LLMs,
registry store, forecaster, conductor, eval/live contexts, API app, doctor
probes) is injectable so the CLI is testable without network or DB (CLAUDE.md
§2.8).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from api.server import DelphiApp
from benchmarks.base import BenchmarkAdapter
from benchmarks.live_loop import ClaimOutcome, claim_and_run
from benchmarks.live_loop.harvest import HarvestJob
from benchmarks.live_loop.score import ScoreJob
from common.doctor import Probe, format_report, run_checks
from conductor.heuristic import HeuristicConductor
from core.orchestration.meta.holdout import HoldoutGovernor
from core.orchestration.run_state import RunStateStore
from core.registry.store import InMemoryRegistryStore, RegistryStore
from evaluation.report import EvalContext, render_leakage_audit, render_report
from forecaster.chain import Forecaster
from intake.llm import StructuredLLM
from intake.service import IntakeService
from resolution.service import ResolutionService

__all__ = ["build_parser", "main"]


@dataclass(frozen=True)
class LiveContext:
    """Everything the ``delphi bench live`` command needs (injected for tests)."""

    harvest_job: HarvestJob
    score_job: ScoreJob
    adapter: BenchmarkAdapter
    run_state: RunStateStore


def build_parser() -> argparse.ArgumentParser:
    """Build the ``delphi`` argument parser (§8 surface)."""
    parser = argparse.ArgumentParser(prog="delphi", description="DELPHI superforecaster CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    intake = sub.add_parser("intake", help="Show the normalized, resolvable form (or refusal).")
    intake.add_argument("question", help="The question to run through intake.")
    intake.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        help="Optional ISO-8601 timestamp enabling the 'already resolved' refusal check.",
    )

    forecast = sub.add_parser("forecast", help="Form a calibrated forecast (writes to registry).")
    forecast.add_argument("question", help="The question to forecast.")
    forecast.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        help="ISO-8601 knowledge-time ceiling the forecast is formed as of (required).",
    )
    forecast.add_argument(
        "--deep",
        action="store_true",
        help="(placeholder) request the deeper ensemble/orchestration tier.",
    )

    resolve = sub.add_parser("resolve", help="Resolve closed questions and write resolutions.")
    resolve.add_argument(
        "--since",
        dest="since",
        default=None,
        help="Optional ISO-8601 timestamp; only resolve questions intook at/after it.",
    )
    resolve_source = resolve.add_mutually_exclusive_group()
    resolve_source.add_argument(
        "--answers",
        dest="answers",
        default=None,
        help="Path to a JSON answer key (question_id -> {value, resolved_at, ...}).",
    )
    resolve_source.add_argument(
        "--suite",
        dest="suite",
        default=None,
        choices=_EVAL_SUITES,
        help="Resolve from a benchmark suite's resolved questions (network).",
    )

    eval_parser = sub.add_parser("eval", help="Proper scores + baselines + CIs + leakage audit.")
    eval_parser.add_argument("--suite", default="default", help="Benchmark suite name.")
    eval_audit = eval_parser.add_mutually_exclusive_group()
    eval_audit.add_argument(
        "--with-leakage-audit",
        dest="with_leakage_audit",
        action="store_true",
        help="Render the score report AND the leakage audit from the same run "
        "(one forecasting pass; §2.6 leakage-first reporting).",
    )
    eval_audit.add_argument(
        "--leakage-audit",
        dest="leakage_audit",
        action="store_true",
        help="Report leakage rate + flagged-at-chance robustness instead of scores.",
    )

    conductor = sub.add_parser(
        "conductor", help="Forecast via the heuristic conductor (records a workflow trace)."
    )
    conductor.add_argument("question", help="The question to forecast.")
    conductor.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        help="ISO-8601 knowledge-time ceiling the forecast is formed as of (required).",
    )

    bench = sub.add_parser("bench", help="Benchmark loops (nightly live harvest/score).")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    live = bench_sub.add_parser("live", help="Run the live loop (harvest or score).")
    mode = live.add_mutually_exclusive_group(required=True)
    mode.add_argument("--harvest", action="store_true", help="Harvest + forecast open questions.")
    mode.add_argument("--score", action="store_true", help="Resolve + score matured questions.")
    live.add_argument(
        "--since", dest="since", default=None, help="(score) only resolve questions since this ts."
    )
    live.add_argument(
        "--tick", dest="tick", default=None, help="ISO-8601 tick id for the idempotent run claim."
    )
    live.add_argument(
        "--suite",
        default="metaculus",
        help="Benchmark suite to harvest/score (metaculus | forecastbench).",
    )

    serve = sub.add_parser("serve", help="Serve the published OpenAI-compatible API.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve.add_argument("--port", type=int, default=8080, help="Bind port.")
    serve.add_argument(
        "--check",
        action="store_true",
        help="Run a health-check round-trip and exit without binding a socket.",
    )

    sub.add_parser(
        "doctor",
        help="Check every external dependency (Postgres, the LLM/Claude API, Tavily, snapshots).",
    )

    return parser


def _parse_as_of(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    parsed = datetime.fromisoformat(raw)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def cmd_intake(args: argparse.Namespace, *, llm: StructuredLLM, store: RegistryStore) -> int:
    """Run a question through intake and print the result."""
    outcome = IntakeService(llm=llm, store=store).intake(
        args.question, as_of=_parse_as_of(args.as_of)
    )
    if outcome.accepted and outcome.resolvable is not None:
        resolvable = outcome.resolvable
        print(f"ACCEPTED question_id={outcome.question_id}")
        print(f"  type: {resolvable.question_type.value}")
        print(f"  domain: {resolvable.domain}")
        print(f"  criteria: {resolvable.resolution_criteria}")
        if resolvable.close_time is not None:
            print(f"  close_time: {resolvable.close_time.isoformat()}")
        return 0
    decision = outcome.refusal
    reason = decision.reason.value if decision and decision.reason else "unknown"
    print(f"REFUSED reason={reason}")
    if decision and decision.detail:
        print(f"  detail: {decision.detail}")
    return 1


def cmd_forecast(args: argparse.Namespace, *, forecaster: Forecaster) -> int:
    """Form a forecast and render probability + band + rationale + provenance."""
    as_of = _parse_as_of(args.as_of)
    assert as_of is not None  # --as-of is required for forecast
    result = forecaster.forecast(args.question, as_of=as_of)
    if not result.accepted or result.probability is None:
        decision = result.refusal
        reason = decision.reason.value if decision and decision.reason else "unknown"
        print(f"REFUSED reason={reason}")
        if decision and decision.detail:
            print(f"  detail: {decision.detail}")
        return 1
    band = result.uncertainty.combined if result.uncertainty is not None else 0.0
    low = max(0.0, result.probability - band)
    high = min(1.0, result.probability + band)
    print(f"FORECAST question_id={result.question_id} forecast_id={result.forecast_id}")
    print(f"  probability: {result.probability:.3f}")
    print(f"  band: [{low:.3f}, {high:.3f}]")
    print(f"  rationale: {result.rationale}")
    for ev in result.evidence:
        print(f"  evidence: [{ev.source_id}] ({ev.knowledge_time.date().isoformat()})")
    if result.quarantined:
        print("  WARNING: quarantined by the leakage judge (post-as-of reference).")
    return 0


def cmd_resolve(args: argparse.Namespace, *, service: ResolutionService) -> int:
    """Resolve open questions and report how many resolutions were written."""
    run = service.resolve_open(since=_parse_as_of(args.since))
    print(f"RESOLVED {len(run.resolved)} question(s); skipped {len(run.skipped)}.")
    for resolution_id in run.resolved:
        print(f"  resolution_id: {resolution_id}")
    return 0


def cmd_doctor(args: argparse.Namespace, *, checks: Sequence[tuple[str, Probe]]) -> int:
    """Run every dependency check and print a PASS/FAIL report (exit 1 on any FAIL)."""
    _ = args
    results = run_checks(checks)
    report, all_ok = format_report(results)
    print(report)
    print("DOCTOR ok" if all_ok else "DOCTOR failed")
    return 0 if all_ok else 1


def cmd_conductor(args: argparse.Namespace, *, conductor: HeuristicConductor) -> int:
    """Forecast via the heuristic conductor and render the workflow trace."""
    as_of = _parse_as_of(args.as_of)
    assert as_of is not None  # --as-of is required for conductor
    result = conductor.conduct(args.question, as_of=as_of)
    forecast = result.forecast
    if not forecast.accepted or forecast.probability is None:
        decision = forecast.refusal
        reason = decision.reason.value if decision and decision.reason else "unknown"
        print(f"REFUSED reason={reason}")
        return 1
    print(f"CONDUCTOR question_id={forecast.question_id} forecast_id={forecast.forecast_id}")
    print(f"  probability: {forecast.probability:.3f}")
    print(f"  route: {' -> '.join(result.workflow.route)}")
    print(f"  revisions: {result.revisions}")
    print(f"  verifier: {'accepted' if result.verifier_accepted else 'quarantined'}")
    print(f"  red-team: {result.red_team_counter}")
    return 0


def cmd_bench_live(args: argparse.Namespace, *, context: LiveContext) -> int:
    """Run one live-loop tick (harvest or score) under an idempotent run claim."""
    tick = _parse_as_of(args.tick)
    if tick is None:  # pragma: no cover - production uses wall-clock tick ids
        tick = datetime.now(UTC)
    mode = "harvest" if args.harvest else "score"
    step_id = f"live-{mode}:{tick.isoformat()}"

    if args.harvest:

        def action() -> object:
            return context.harvest_job.run(context.adapter)
    else:
        since = _parse_as_of(args.since)

        def action() -> object:
            return context.score_job.run(since=since)

    outcome, result = claim_and_run(context.run_state, step_id=step_id, tick_at=tick, action=action)
    if outcome == ClaimOutcome.SKIPPED:
        print(f"SKIPPED live {mode} (already succeeded for this tick).")
        return 0
    if args.harvest:
        from benchmarks.live_loop.harvest import HarvestRun

        assert isinstance(result, HarvestRun)
        print(f"HARVEST pending={result.count} refused={len(result.refused)}")
    else:
        from benchmarks.live_loop.score import ScoreRun

        assert isinstance(result, ScoreRun)
        metrics = result.metrics
        brier = "n/a" if metrics.brier is None else f"{metrics.brier:.4f}"
        print(f"SCORE resolved={len(result.resolved)} n={metrics.n} brier={brier}")
    return 0


def cmd_serve(args: argparse.Namespace, *, app: DelphiApp) -> int:
    """Serve the published API, or (with ``--check``) verify health and exit."""
    status, _payload = app.handle("GET", "/healthz")
    print(f"DELPHI API health={status} host={args.host} port={args.port}")
    if not app.auth_enabled:
        print(
            "WARNING: DELPHI_SECRET_API_TOKEN not set - serving an UNAUTHENTICATED "
            "endpoint (local development only; api/wsgi.py refuses to start without it)."
        )
    if args.check:
        return 0 if status == 200 else 1
    from api.server import serve  # pragma: no cover - binds a socket

    serve(app, host=args.host, port=args.port)  # pragma: no cover - blocks
    return 0  # pragma: no cover - unreachable while serving


def cmd_eval(args: argparse.Namespace, *, context: EvalContext) -> int:
    """Score a suite (or run a leakage audit) and print the report."""
    judge = context.judge
    if (args.leakage_audit or args.with_leakage_audit) and judge is None:
        print("No leakage judge configured for this suite.")
        return 1
    if args.leakage_audit:
        assert judge is not None  # guarded above
        print(render_leakage_audit(judge, context.inputs.traces))
        return 0
    print(render_report(context.inputs, harness=context.harness))
    if args.with_leakage_audit:
        assert judge is not None  # guarded above
        print(render_leakage_audit(judge, context.inputs.traces))
    return 0


def _default_llm() -> StructuredLLM:  # pragma: no cover - requires network + API key
    from common.llm import structured_client_for_tier
    from common.settings import load_settings

    # Provider chosen by DELPHI_LLM_PROVIDER (default: direct Anthropic API).
    return structured_client_for_tier(load_settings(), "opus")


def _default_store() -> RegistryStore:  # pragma: no cover - requires Postgres
    from common.settings import load_settings
    from core.registry.store import PostgresRegistryStore

    settings = load_settings()
    if settings.pg_dsn:
        return PostgresRegistryStore.connect(settings.pg_dsn)
    return InMemoryRegistryStore()


_DEFAULT_SNAPSHOT_DIR = "~/.delphi/snapshots"


def _snapshot_store(snapshot_dir: str | None) -> Any:  # pragma: no cover - filesystem side effect
    """Durable file-backed snapshot store so real retrieval is reproducible."""
    from pathlib import Path

    from sources.snapshot import FileSnapshotStore

    root = Path(snapshot_dir or _DEFAULT_SNAPSHOT_DIR).expanduser()
    return FileSnapshotStore(root)


_EVIDENCE_PROVIDERS = ("tavily", "gdelt", "wikipedia")

# Wikimedia and GDELT both require polite clients: a descriptive User-Agent
# (Wikimedia robot policy returns 403 without one) and modest pacing (GDELT's
# keyless API 429s under bursts).
_EVIDENCE_USER_AGENT = "DELPHI-forecaster/0.1 (+https://github.com/jzyzak/delphi)"


def _evidence_provider_names(env: dict[str, str] | None = None) -> tuple[str, ...]:
    """Parse DELPHI_EVIDENCE_PROVIDERS (comma-separated; default: tavily).

    Order is preserved and duplicates removed; unknown names raise so a typo
    cannot silently drop an evidence source.
    """
    import os

    e = env if env is not None else dict(os.environ)
    raw = e.get("DELPHI_EVIDENCE_PROVIDERS") or "tavily"
    names: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in _EVIDENCE_PROVIDERS:
            valid = ", ".join(_EVIDENCE_PROVIDERS)
            msg = f"unknown evidence provider {name!r}; choose from: {valid}."
            raise ValueError(msg)
        if name not in names:
            names.append(name)
    if not names:
        msg = "DELPHI_EVIDENCE_PROVIDERS must name at least one provider."
        raise ValueError(msg)
    return tuple(names)


def _default_forecaster() -> Forecaster:  # pragma: no cover - requires LLM API + hosted search
    from common.composition import build_postgres_composition
    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from common.llm import LLMConfig
    from common.secrets import EnvSecretProvider
    from core.forecast.bayesian import BedrockEvidenceLikelihoodLLM
    from core.forecast.leakage_judge import BedrockLeakageJudgeLLM, LeakageJudge
    from core.forecast.llm import BedrockForecastLLM
    from core.forecast.supervisor import BedrockSupervisorLLM, Confidence
    from forecaster.stages.aggregate import SupervisorTuning
    from sources.providers.tavily import TavilySearchClient, tavily_config

    comp = build_postgres_composition()
    settings = comp.settings
    # Reasoning-grade config: adaptive thinking + high effort + headroom for
    # the reasoning tokens (anthropic transport only; see common/llm/config.py).
    reasoning_cfg = LLMConfig(thinking="adaptive", effort="high", max_tokens=4096)
    reasoning = comp.structured_client("opus", config=reasoning_cfg)
    http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
    snapshot_store = _snapshot_store(settings.snapshot_dir)
    provider_names = _evidence_provider_names()
    searchers = []
    for name in provider_names:
        if name == "tavily":
            tavily = TavilySearchClient(
                http=http, config=tavily_config(), secrets=EnvSecretProvider()
            )
            searchers.append(
                comp.hosted_searcher(http_client=http, client=tavily, snapshot_store=snapshot_store)
            )
        elif name == "gdelt":
            from sources.providers.gdelt import GdeltAsOfSearcher

            gdelt_http = HttpClient(
                config=HttpConfig(
                    user_agent=settings.http_user_agent or _EVIDENCE_USER_AGENT,
                    min_interval_s=6.0,
                    # A 429 from GDELT means an IP cooldown: retrying deepens
                    # it. One attempt; the composite skips the provider.
                    max_retries=1,
                )
            )
            searchers.append(GdeltAsOfSearcher(http=gdelt_http, snapshot_store=snapshot_store))
        elif name == "wikipedia":
            from sources.providers.wikipedia import WikipediaAsOfSearcher

            wiki_http = HttpClient(
                config=HttpConfig(
                    user_agent=settings.http_user_agent or _EVIDENCE_USER_AGENT,
                    min_interval_s=1.0,
                )
            )
            searchers.append(WikipediaAsOfSearcher(http=wiki_http, snapshot_store=snapshot_store))
    if len(searchers) == 1:
        searcher = searchers[0]
    else:
        from sources.searcher import CompositeAsOfSearcher

        searcher = CompositeAsOfSearcher(searchers)
    return Forecaster(
        intake=IntakeService(llm=reasoning, store=comp.registry_store),
        searcher=searcher,
        reasoning_llm=reasoning,
        forecast_llm=BedrockForecastLLM(comp.structured_client("opus", config=reasoning_cfg)),
        supervisor_llm=BedrockSupervisorLLM(comp.structured_client("fable", config=reasoning_cfg)),
        leakage_judge=LeakageJudge(BedrockLeakageJudgeLLM(comp.structured_client("opus"))),
        registry_store=comp.registry_store,
        # Bayesian path: prior = reference-class base rate, per-draw evidence
        # log-LRs combined in log-odds space (fights no-evidence 0.5 collapse).
        evidence_likelihood_llm=BedrockEvidenceLikelihoodLLM(
            comp.structured_client("opus", config=reasoning_cfg)
        ),
        bayesian_draws=12,
        runs_per_agent=3,
        aggregator="trimmed_mean",
        # Looser trigger + MEDIUM apply gate: a 12-draw ensemble has meaningful
        # spread, and the supervisor may improve on the aggregate more often.
        supervisor_tuning=SupervisorTuning(
            spread_threshold=0.08, min_apply_confidence=Confidence.MEDIUM
        ),
    )


_EVAL_SUITES = ("metaculus", "forecastbench")


def _holdout_governor_from_env(
    env: dict[str, str] | None = None,
) -> HoldoutGovernor | None:
    """Optional guarded-holdout wiring for ``delphi eval`` (CLAUDE.md §2.2).

    Fail-closed by default: returns ``None`` (the harness then refuses every
    holdout access) unless BOTH ``DELPHI_HOLDOUT_FILE`` (path to a JSON payload)
    and ``DELPHI_HOLDOUT_BUDGET`` (max logged accesses) are set. Every access
    through the returned governor is hash-chain logged and debits the budget.
    """
    import json
    import os
    from pathlib import Path

    from core.orchestration.meta.holdout import (
        InMemoryHoldoutGovernor,
        StaticHoldoutSource,
    )

    e = env if env is not None else dict(os.environ)
    file_path = e.get("DELPHI_HOLDOUT_FILE")
    budget_raw = e.get("DELPHI_HOLDOUT_BUDGET")
    if not file_path or not budget_raw:
        return None
    try:
        budget = int(budget_raw)
    except ValueError as exc:
        msg = f"DELPHI_HOLDOUT_BUDGET must be an integer, got {budget_raw!r}."
        raise ValueError(msg) from exc
    payload = json.loads(Path(file_path).read_text())
    if not isinstance(payload, dict):
        msg = f"DELPHI_HOLDOUT_FILE must contain a JSON object, got {type(payload).__name__}."
        raise ValueError(msg)
    return InMemoryHoldoutGovernor(budget=budget, source=StaticHoldoutSource(payload))


def _eval_record_limits(
    env: dict[str, str] | None = None,
) -> tuple[str | None, int | None, int]:
    """Read the retrospective-eval sampling knobs from the environment.

    Returns ``(resolved_after, max_questions, max_pages)``. ``resolved_after``
    (``DELPHI_EVAL_RESOLVED_AFTER``, ISO-8601) guards against scoring questions
    the models may have memorized (§2.6: resolutions predating the model
    training cutoff are suspect). ``max_questions``
    (``DELPHI_EVAL_MAX_QUESTIONS``) bounds LLM spend. ``max_pages``
    (``DELPHI_EVAL_MAX_PAGES``, metaculus only) controls fetch depth
    (default 5).
    """
    import os

    e = env if env is not None else dict(os.environ)
    resolved_after = e.get("DELPHI_EVAL_RESOLVED_AFTER") or None
    raw_max = e.get("DELPHI_EVAL_MAX_QUESTIONS")
    raw_pages = e.get("DELPHI_EVAL_MAX_PAGES")
    try:
        max_questions = int(raw_max) if raw_max else None
        max_pages = int(raw_pages) if raw_pages else 5
    except ValueError as exc:
        msg = "DELPHI_EVAL_MAX_QUESTIONS / DELPHI_EVAL_MAX_PAGES must be integers."
        raise ValueError(msg) from exc
    return resolved_after, max_questions, max_pages


def _filter_eval_records(
    records: Sequence[dict[str, Any]],
    *,
    resolved_after: str | None,
    max_questions: int | None,
) -> list[dict[str, Any]]:
    """Bound a retrospective eval's question set before any forecasting spend.

    ``resolved_after`` keeps only records whose ``resolved_at`` is strictly
    later than the cutoff (records with no ``resolved_at`` are dropped — they
    cannot be scored). ``max_questions`` keeps the freshest N by
    ``resolved_at`` descending (deterministic: freshest first, ties broken by
    the fetch order being stable under ``list.sort``).
    """
    from benchmarks.base import parse_dt

    out = list(records)
    if resolved_after is not None:
        cutoff = parse_dt(resolved_after)
        out = [
            record
            for record in out
            if record.get("resolved_at") is not None and parse_dt(record["resolved_at"]) > cutoff
        ]
    if max_questions is not None:
        floor = datetime.min.replace(tzinfo=UTC)

        def freshness(record: dict[str, Any]) -> datetime:
            raw = record.get("resolved_at")
            return parse_dt(raw) if raw is not None else floor

        out.sort(key=freshness, reverse=True)
        out = out[:max_questions]
    return out


def _default_eval_context(suite: str) -> EvalContext:  # pragma: no cover - network + LLM + DB
    """Wire a retrospective evaluation suite (fetch -> adapter -> forecast -> score).

    Requires network (benchmark fetch + hosted search), the LLM tiers, and the
    registry/ledger DB, so it is exercised via the injected ``eval_context`` in
    tests rather than here. The trials ledger is drawn against per §2.4.
    """
    import os

    from benchmarks.fetchers import ForecastBenchFetcher, MetaculusFetcher
    from benchmarks.forecastbench import ForecastBenchAdapter
    from benchmarks.market_consensus import consensus_baseline
    from benchmarks.metaculus import MetaculusAdapter
    from benchmarks.suites import build_eval_context, forecaster_fn, records_baseline
    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from common.secrets import EnvSecretProvider
    from common.settings import load_settings
    from core.forecast.leakage_judge import BedrockLeakageJudgeLLM, LeakageJudge
    from core.orchestration.budget import InMemoryBudgetLedger, PostgresBudgetLedger
    from evaluation.baselines import Baseline
    from evaluation.harness import EvalHarness

    settings = load_settings()
    http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
    forecast_fn = forecaster_fn(_default_forecaster())

    baselines: list[Baseline] = []
    resolved_after, max_questions, max_pages = _eval_record_limits()
    if suite == "metaculus":
        records = MetaculusFetcher(http=http, secrets=EnvSecretProvider()).fetch(
            params={"statuses": "resolved", "forecast_type": "binary", "limit": 100},
            max_pages=max_pages,
        )
        records = _filter_eval_records(
            records, resolved_after=resolved_after, max_questions=max_questions
        )
        adapter: BenchmarkAdapter = MetaculusAdapter.from_records(records)
        baselines.append(consensus_baseline(adapter, price_key="community_prediction"))
    elif suite == "forecastbench":
        question_set = os.environ["DELPHI_FORECASTBENCH_QUESTION_SET"]
        resolution_set = os.environ.get("DELPHI_FORECASTBENCH_RESOLUTION_SET")
        records = ForecastBenchFetcher(http=http).fetch(
            question_set=question_set, resolution_set=resolution_set
        )
        records = _filter_eval_records(
            records, resolved_after=resolved_after, max_questions=max_questions
        )
        adapter = ForecastBenchAdapter.from_records(records)
        baselines.append(records_baseline(records, source="forecastbench"))
    else:
        valid = ", ".join(_EVAL_SUITES)
        msg = f"unknown --suite {suite!r}; choose one of: {valid}."
        raise ValueError(msg)

    from common.composition import build_postgres_composition

    comp = build_postgres_composition()
    judge = LeakageJudge(BedrockLeakageJudgeLLM(comp.structured_client("opus")))

    cap = settings.global_trials_budget
    ledger = (
        PostgresBudgetLedger.connect(settings.pg_dsn, cap=cap)
        if settings.pg_dsn
        else InMemoryBudgetLedger(cap=cap, trials_count=lambda: 0)
    )
    return build_eval_context(
        adapter,
        forecast_fn,
        harness=EvalHarness(budget_ledger=ledger, holdout=_holdout_governor_from_env()),
        judge=judge,
        extra_baselines=tuple(baselines),
    )


def _default_resolution_service(
    answers: str | None = None,
    *,
    suite: str | None = None,
) -> ResolutionService:  # pragma: no cover - requires Postgres (and network for --suite)
    from resolution.benchmark_source import BenchmarkResolutionSource
    from resolution.sources import (
        MappingResolutionSource,
        ResolutionSource,
        load_mapping_source,
    )

    source: ResolutionSource
    if suite is not None:
        source = BenchmarkResolutionSource(_fetch_benchmark_resolutions(suite))
    elif answers:
        source = load_mapping_source(answers)
    else:
        source = MappingResolutionSource({})
    return ResolutionService(store=_default_store(), source=source)


def _fetch_benchmark_resolutions(suite: str):  # pragma: no cover - network
    """Fetch a suite's resolved questions as registry resolutions (same wiring
    as the live loop's scoring phase in ``_default_live_context``)."""
    import os

    from benchmarks.fetchers import ForecastBenchFetcher, MetaculusFetcher
    from benchmarks.forecastbench import ForecastBenchAdapter
    from benchmarks.metaculus import MetaculusAdapter
    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from common.secrets import EnvSecretProvider
    from common.settings import load_settings

    settings = load_settings()
    http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
    if suite == "metaculus":
        records = MetaculusFetcher(http=http, secrets=EnvSecretProvider()).fetch(
            params={"statuses": "resolved", "forecast_type": "binary", "limit": 100},
            max_pages=5,
        )
        return MetaculusAdapter.from_records(records).resolutions()
    if suite == "forecastbench":
        question_set = os.environ["DELPHI_FORECASTBENCH_QUESTION_SET"]
        resolution_set = os.environ.get("DELPHI_FORECASTBENCH_RESOLUTION_SET")
        if resolution_set is None:
            return ()
        fb = ForecastBenchFetcher(http=http)
        records = fb.fetch(question_set=question_set, resolution_set=resolution_set)
        return ForecastBenchAdapter.from_records(records).resolutions()
    valid = ", ".join(_EVAL_SUITES)
    msg = f"unknown --suite {suite!r}; choose one of: {valid}."
    raise ValueError(msg)


def _default_doctor_checks() -> list[tuple[str, Probe]]:  # pragma: no cover - probes hit infra
    """Build the real dependency probes for ``delphi doctor``."""
    from pathlib import Path

    from common.composition import build_postgres_composition
    from common.http.client import HttpClient
    from common.secrets import EnvSecretProvider
    from sources.providers.tavily import TavilySearchClient, tavily_config

    def check_postgres() -> str:
        comp = build_postgres_composition()
        n = len(comp.registry_store.all_questions())
        return f"connected + migrated; {n} question(s) recorded."

    def check_llm() -> str:
        from common.llm.tiering import structured_client_for_tier
        from common.settings import load_settings

        settings = load_settings()
        details = []
        for tier in ("opus", "fable"):
            client = structured_client_for_tier(settings, tier)
            client.invoke_structured(
                system="Reply only with compact JSON.",
                user='Return the JSON object {"ok": true}.',
            )
            details.append(f"{tier}={settings.model_for_tier(tier)}")
        return f"{settings.llm_provider} reachable: " + ", ".join(details)

    def check_tavily() -> str:
        http = HttpClient()
        client = TavilySearchClient(http=http, config=tavily_config(), secrets=EnvSecretProvider())
        response = client.search("test query", max_results=1)
        return f"reachable; {len(response.results)} result(s) for a probe query."

    def check_snapshots() -> str:
        from common.settings import load_settings

        root = Path(load_settings().snapshot_dir or _DEFAULT_SNAPSHOT_DIR).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".delphi_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return f"writable at {root}."

    return [
        ("postgres", check_postgres),
        ("llm", check_llm),
        ("tavily", check_tavily),
        ("snapshots", check_snapshots),
    ]


def _default_conductor() -> HeuristicConductor:  # pragma: no cover - requires LLM API + search
    return HeuristicConductor(forecaster=_default_forecaster())


def _default_live_context(suite: str) -> LiveContext:  # pragma: no cover - network + LLM + DB
    """Wire the nightly live loop for a benchmark suite.

    Harvest pins every open question's as-of to the harvest instant (via the
    fetcher's ``freeze_at``) and skips questions already forecast (dedup against
    the registry). Scoring resolves matured questions through a
    :class:`BenchmarkResolutionSource` keyed on the benchmark id threaded into
    each question's metadata at harvest time. Requires network + LLM + DB, so it
    is exercised via the injected ``live_context`` in tests.
    """
    import os

    from benchmarks.fetchers import ForecastBenchFetcher, MetaculusFetcher
    from benchmarks.forecastbench import ForecastBenchAdapter
    from benchmarks.metaculus import MetaculusAdapter
    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from common.secrets import EnvSecretProvider
    from common.settings import load_settings
    from core.orchestration.run_state import InMemoryRunStateStore, PostgresRunStateStore
    from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY, BenchmarkResolutionSource

    settings = load_settings()
    http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
    harvest_time = datetime.now(UTC)
    store = _default_store()
    seen = {
        q.metadata.get(BENCHMARK_QUESTION_ID_KEY)
        for q in store.all_questions()
        if isinstance(q.metadata.get(BENCHMARK_QUESTION_ID_KEY), str)
    }

    if suite == "metaculus":
        fetcher = MetaculusFetcher(http=http, secrets=EnvSecretProvider())
        open_records = [
            record
            for record in fetcher.fetch(
                params={"statuses": "open", "forecast_type": "binary", "limit": 100},
                max_pages=5,
                freeze_at=harvest_time,
            )
            if f"metaculus:{record['id']}" not in seen
        ]
        harvest_adapter: BenchmarkAdapter = MetaculusAdapter.from_records(open_records)
        resolved = fetcher.fetch(
            params={"statuses": "resolved", "forecast_type": "binary", "limit": 100},
            max_pages=5,
        )
        resolutions = MetaculusAdapter.from_records(resolved).resolutions()
    elif suite == "forecastbench":
        fb = ForecastBenchFetcher(http=http)
        question_set = os.environ["DELPHI_FORECASTBENCH_QUESTION_SET"]
        resolution_set = os.environ.get("DELPHI_FORECASTBENCH_RESOLUTION_SET")
        open_records = [
            record
            for record in fb.fetch(question_set=question_set, freeze_at=harvest_time)
            if f"forecastbench:{record['id']}" not in seen
        ]
        harvest_adapter = ForecastBenchAdapter.from_records(open_records)
        resolutions = ()
        if resolution_set is not None:
            resolved = fb.fetch(question_set=question_set, resolution_set=resolution_set)
            resolutions = ForecastBenchAdapter.from_records(resolved).resolutions()
    else:
        valid = ", ".join(_EVAL_SUITES)
        msg = f"unknown --suite {suite!r}; choose one of: {valid}."
        raise ValueError(msg)

    run_state: RunStateStore = (
        PostgresRunStateStore.connect(settings.pg_dsn)
        if settings.pg_dsn
        else InMemoryRunStateStore()
    )
    return LiveContext(
        harvest_job=HarvestJob(conductor=_default_conductor()),
        score_job=ScoreJob(
            store=store,
            resolution_service=ResolutionService(
                store=store, source=BenchmarkResolutionSource(resolutions)
            ),
        ),
        adapter=harvest_adapter,
        run_state=run_state,
    )


def _default_api_app(
    auth_token: str | None = None,
) -> DelphiApp:  # pragma: no cover - requires LLM API + hosted search
    """Wire the production API app.

    Bearer auth is enforced when ``auth_token`` is given, or (for local
    ``delphi serve``) when ``DELPHI_SECRET_API_TOKEN`` is set in the environment.
    The production entry point (``api.wsgi``) always passes a token explicitly.
    """
    import os

    from api.routes import ForecastService

    token = auth_token or os.environ.get("DELPHI_SECRET_API_TOKEN") or None
    forecaster = _default_forecaster()
    conductor = HeuristicConductor(forecaster=forecaster)
    service = ForecastService(forecaster=forecaster, conductor=conductor, store=_default_store())
    return DelphiApp(service, auth_token=token)


def main(
    argv: list[str] | None = None,
    *,
    llm: StructuredLLM | None = None,
    store: RegistryStore | None = None,
    forecaster: Forecaster | None = None,
    resolution_service: ResolutionService | None = None,
    eval_context: EvalContext | None = None,
    conductor: HeuristicConductor | None = None,
    live_context: LiveContext | None = None,
    api_app: DelphiApp | None = None,
    doctor_checks: Sequence[tuple[str, Probe]] | None = None,
) -> int:
    """Entry point. Dependencies may be injected for tests (no network/DB)."""
    args = build_parser().parse_args(argv)
    if args.command == "intake":
        return cmd_intake(
            args,
            llm=llm if llm is not None else _default_llm(),
            store=store if store is not None else _default_store(),
        )
    if args.command == "forecast":
        return cmd_forecast(
            args,
            forecaster=forecaster if forecaster is not None else _default_forecaster(),
        )
    if args.command == "resolve":
        if resolution_service is None and args.answers is None and args.suite is None:
            print(
                "Nothing to resolve from: pass --answers FILE or --suite "
                + "|".join(_EVAL_SUITES)
                + "."
            )
            return 2
        return cmd_resolve(
            args,
            service=resolution_service
            if resolution_service is not None
            else _default_resolution_service(args.answers, suite=args.suite),
        )
    if args.command == "eval":
        if eval_context is None:
            eval_context = _default_eval_context(args.suite)  # pragma: no cover - wired suite
        return cmd_eval(args, context=eval_context)
    if args.command == "conductor":
        return cmd_conductor(
            args, conductor=conductor if conductor is not None else _default_conductor()
        )
    if args.command == "bench" and args.bench_command == "live":
        if live_context is None:
            live_context = _default_live_context(args.suite)  # pragma: no cover - wired source
        return cmd_bench_live(args, context=live_context)
    if args.command == "serve":
        return cmd_serve(args, app=api_app if api_app is not None else _default_api_app())
    if args.command == "doctor":
        return cmd_doctor(
            args,
            checks=doctor_checks if doctor_checks is not None else _default_doctor_checks(),
        )
    # Unreachable: argparse rejects unknown commands before we get here.
    print(f"'{args.command}' is not implemented yet.")  # pragma: no cover
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover - module execution guard
    raise SystemExit(main())
