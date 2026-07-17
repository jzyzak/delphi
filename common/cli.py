"""``delphi`` command-line entry point.

The full §8 CLI surface: ``intake``, ``forecast``, ``resolve``, ``eval``,
``conductor``, ``bench live``, ``serve``, and ``doctor``. Every dependency (LLMs,
registry store, forecaster, conductor, eval/live contexts, API app, doctor
probes) is injectable so the CLI is testable without network or DB (CLAUDE.md
§2.8).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.server import DelphiApp
from benchmarks.base import BenchmarkAdapter
from benchmarks.live_loop import ClaimOutcome, claim_and_run
from benchmarks.live_loop.harvest import HarvestJob
from benchmarks.live_loop.score import ScoreJob
from common.doctor import Probe, format_report, run_checks
from conductor.heuristic import HeuristicConductor
from core.orchestration.budget import BudgetLedger
from core.orchestration.meta.holdout import HoldoutGovernor
from core.orchestration.run_state import RunStateStore
from core.registry.store import InMemoryRegistryStore, RegistryStore
from evaluation.baselines import Baseline
from evaluation.calibration_split import (
    assign_calibration_split,
    fit_calibration_artifact,
    question_fingerprint,
)
from evaluation.report import EvalContext, render_leakage_audit, render_report
from evaluation.scoring import ScoredRecord
from forecaster.calibration_artifact import artifact_filename
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
    eval_parser.add_argument(
        "--calibration-artifact",
        dest="calibration_artifact",
        default=None,
        help=(
            "Path to a pre-fitted calibration artifact (from `delphi calibration "
            "fit` on a disjoint corpus): applied to ALL questions, all scored."
        ),
    )
    eval_parser.add_argument(
        "--no-search",
        dest="no_search",
        action="store_true",
        help="Ablation arm: forecast with an empty evidence searcher (no retrieval).",
    )
    eval_parser.add_argument(
        "--only-sources",
        dest="only_sources",
        default=None,
        metavar="SRC[,SRC...]",
        help=(
            "forecastbench only: restrict to question families by composite-id "
            "prefix (e.g. fred,dbnomics) for a targeted domain eval."
        ),
    )
    eval_parser.add_argument(
        "--exclude-fit-questions",
        dest="exclude_fit_questions",
        action="store_true",
        help=(
            "Artifact mode: drop questions in the artifact's fit set BEFORE "
            "forecasting (never scored, never charged; §2.5) instead of "
            "refusing the whole run. The exclusion count is reported."
        ),
    )
    eval_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Stratified deterministic subsample of N questions (see --sample-seed).",
    )
    eval_parser.add_argument(
        "--sample-seed",
        dest="sample_seed",
        type=int,
        default=0,
        help="Seed for --sample; identical (N, seed) yields identical questions.",
    )
    eval_parser.add_argument(
        "--dump-forecasts",
        dest="dump_forecasts",
        default=None,
        help="Write the scored {question_id: probability} JSON to this path.",
    )
    eval_parser.add_argument(
        "--extra-baseline",
        dest="extra_baselines",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add a baseline from a dumped forecasts JSON (repeatable).",
    )
    eval_parser.add_argument(
        "--resume-file",
        dest="resume_file",
        default=None,
        help=(
            "JSONL progress log: forecasts persist as they complete, and an "
            "interrupted run rerun with the same file resumes instead of "
            "re-forecasting. Use a fresh file per run configuration."
        ),
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

    calibration = sub.add_parser(
        "calibration", help="Fit / manage the learned calibration artifact (§2.5)."
    )
    calibration_sub = calibration.add_subparsers(dest="calibration_command", required=True)
    fit = calibration_sub.add_parser(
        "fit",
        help="Fit recalibrator + alpha + floor on the disjoint calibration split.",
    )
    fit.add_argument(
        "--output",
        default=".",
        help="Artifact output path; a directory gets a versioned calibration-<date>-<hash>.json.",
    )
    fit.add_argument(
        "--method",
        default="auto",
        choices=("auto", "isotonic", "platt"),
        help="Recalibrator family ('auto' selects by K-fold CV within the split).",
    )
    fit.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of eligible (non-reserved) resolved questions to fit on.",
    )
    fit.add_argument("--seed", type=int, default=0, help="Split/CV seed (deterministic).")
    fit.add_argument(
        "--holdout-file",
        dest="holdout_file",
        default=None,
        help="Path to a JSON array of question ids to exclude (the guarded set).",
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


def _resolved_forecast_records(store: RegistryStore) -> list[ScoredRecord]:
    """Collect (raw probability, outcome) pairs for every resolved binary question.

    Uses the *raw* (pre-recalibration) probability recorded in the forecast's
    calibration metadata so the fit maps raw ensemble output to outcomes — the
    same composition the chain applies at inference time.
    """
    records: list[ScoredRecord] = []
    for question in store.all_questions():
        resolutions = [
            r for r in store.resolutions_for(question.question_id) if r.resolved_value in (0.0, 1.0)
        ]
        if not resolutions:
            continue
        resolution = max(resolutions, key=lambda r: r.resolved_at)
        forecasts = store.forecasts_for(question.question_id)
        if not forecasts:
            continue
        forecast = max(forecasts, key=lambda f: f.as_of)
        raw = forecast.calibration_metadata.get("raw_probability", forecast.probability)
        if raw is None:
            continue
        records.append(
            ScoredRecord(
                question_id=question.question_id,
                domain=question.domain,
                probability=float(raw),
                outcome=float(resolution.resolved_value),
            )
        )
    return records


def _load_holdout_ids(path: str) -> frozenset[str]:
    """Read a JSON array of question ids to exclude from the fit (the guarded set)."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(v, str) for v in data):
        msg = f"holdout file {path} must contain a JSON array of question-id strings."
        raise ValueError(msg)
    return frozenset(data)


def _fit_question_fingerprints(
    store: RegistryStore, split_records: Sequence[ScoredRecord]
) -> list[str]:
    """Fingerprint every fitted question: registry id + text + benchmark id.

    The benchmark identity is taken from the question's metadata AND from any
    resolution's ``resolved_label`` — backfilled resolutions carry the
    benchmark id there when the question predates benchmark-id metadata, and a
    fit question whose benchmark id escapes the artifact could silently be
    re-scored by a later artifact-mode eval (§2.5).
    """
    from resolution.benchmark_source import BENCHMARK_QUESTION_ID_KEY

    fingerprints: set[str] = set()
    for record in split_records:
        fingerprints.add(question_fingerprint(record.question_id))
        question = store.get_question(record.question_id)
        fingerprints.add(question_fingerprint(question.text))
        benchmark_id = question.metadata.get(BENCHMARK_QUESTION_ID_KEY)
        if benchmark_id:
            fingerprints.add(question_fingerprint(str(benchmark_id)))
        for resolution in store.resolutions_for(record.question_id):
            if resolution.resolved_label:
                fingerprints.add(question_fingerprint(resolution.resolved_label))
    return sorted(fingerprints)


def cmd_calibration_fit(
    args: argparse.Namespace,
    *,
    store: RegistryStore,
    clock: Callable[[], datetime] | None = None,
) -> int:
    """Fit the calibration artifact on the disjoint calibration split (§2.5)."""
    try:
        holdout_ids = _load_holdout_ids(args.holdout_file) if args.holdout_file else frozenset()
    except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
        print(f"Cannot read holdout file: {exc}")
        return 1
    records = _resolved_forecast_records(store)
    split = assign_calibration_split(
        [r.question_id for r in records],
        holdout_ids=holdout_ids,
        fraction=args.fraction,
        seed=args.seed,
    )
    split_records = [r for r in records if r.question_id in split]
    if not split_records:
        print(
            "No resolved forecasts available in the calibration split; nothing to fit. "
            "Resolve some questions first (`delphi resolve`)."
        )
        return 1
    artifact = fit_calibration_artifact(split_records, method=args.method, seed=args.seed)
    now = clock() if clock is not None else datetime.now(UTC)
    data = artifact.to_dict()
    data["fitted"] = {
        "fitted_at": now.isoformat(),
        "n": len(split_records),
        "seed": args.seed,
        "fraction": args.fraction,
        "n_holdout_excluded": len(holdout_ids),
        "method_requested": args.method,
        # §2.5 disjointness: artifact-mode eval refuses to score any question
        # that fingerprints into the fit set (registry id, normalized text,
        # or the originating benchmark question id).
        "question_fingerprints": _fit_question_fingerprints(store, split_records),
    }
    if artifact.fallback:
        print(
            f"WARNING: only {len(split_records)} fit point(s) — below the trust "
            "threshold. Wrote the labeled IDENTITY FALLBACK artifact (raw "
            "pass-through); grow the calibration corpus and refit."
        )
    out = Path(args.output).expanduser()
    path = out / artifact_filename(data, date=now.strftime("%Y%m%d")) if out.is_dir() else out
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    floor_text = "none" if artifact.floor is None else f"{artifact.floor:g}"
    print(
        f"Fitted {artifact.recalibrator.method} recalibrator on {len(split_records)} "
        f"resolved question(s): alpha={artifact.alpha:g}, floor={floor_text}."
    )
    print(f"Artifact written to {path}")
    print(f"Export DELPHI_CALIBRATION_ARTIFACT={path} to use it in the live chain.")
    return 0


def _load_extra_baselines(specs: Sequence[str]) -> list[Baseline]:
    """Parse ``NAME=PATH`` specs into baselines from dumped forecast JSONs."""
    baselines: list[Baseline] = []
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name.strip() or not path.strip():
            msg = f"--extra-baseline expects NAME=PATH, got {spec!r}"
            raise ValueError(msg)
        data = json.loads(Path(path.strip()).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            msg = f"baseline file {path!r} must be a JSON object of question_id -> probability."
            raise ValueError(msg)
        predictions = {str(qid): float(p) for qid, p in data.items()}
        baselines.append(Baseline(name=name.strip(), predictions=predictions))
    return baselines


def cmd_eval(args: argparse.Namespace, *, context: EvalContext) -> int:
    """Score a suite and print the report (leakage audit always included, §2.6).

    ``--leakage-audit`` renders the audit alone (no scores, no ledger draw);
    the default scored report carries its own leakage section regardless.
    """
    if args.leakage_audit:
        if context.judge is None:
            print("No leakage judge configured for this suite.")
            return 1
        print(render_leakage_audit(context.judge, context.inputs.traces))
        return 0
    inputs = context.inputs
    extra_specs = getattr(args, "extra_baselines", None) or []
    if extra_specs:
        try:
            extra = _load_extra_baselines(extra_specs)
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            print(f"Cannot load extra baseline: {exc}")
            return 1
        inputs = replace(inputs, baselines=(*inputs.baselines, *extra))
    dump_path = getattr(args, "dump_forecasts", None)
    if dump_path:
        forecasts = {r.question_id: r.probability for r in inputs.records}
        target = Path(dump_path).expanduser()
        target.write_text(json.dumps(forecasts, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Dumped {len(forecasts)} forecast(s) to {target}")
    print(render_report(inputs, harness=context.harness, judge=context.judge))
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
_DEFAULT_CORPUS_PATH = "~/.delphi/corpus.jsonl"


def _snapshot_store(snapshot_dir: str | None) -> Any:  # pragma: no cover - filesystem side effect
    """Durable file-backed snapshot store so real retrieval is reproducible."""
    from pathlib import Path

    from sources.snapshot import FileSnapshotStore

    root = Path(snapshot_dir or _DEFAULT_SNAPSHOT_DIR).expanduser()
    return FileSnapshotStore(root)


_EVIDENCE_PROVIDERS = ("tavily", "gdelt", "wikipedia", "none")
_DEFAULT_EVIDENCE_PROVIDERS = "tavily,gdelt,wikipedia"
# Wikimedia's robot policy requires a descriptive User-Agent WITH a contact
# point (URL or email) — a bare product string gets a 403.
_EVIDENCE_USER_AGENT = "DELPHI-forecaster/0.1 (+https://github.com/jzyzak/delphi)"


def _evidence_provider_names(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Parse DELPHI_EVIDENCE_PROVIDERS (comma-separated; default: all providers).

    Unknown names raise; duplicates collapse (order-preserving). Tavily serves
    current news, GDELT and Wikipedia serve *historical* as-of evidence —
    retrospective benchmarks are meaningless without the latter two. ``none``
    is the explicit no-search arm for ablations and must stand alone.
    """
    import os

    e = os.environ if env is None else env
    raw = e.get("DELPHI_EVIDENCE_PROVIDERS") or _DEFAULT_EVIDENCE_PROVIDERS
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
    if "none" in names and len(names) > 1:
        msg = "'none' (the no-search ablation arm) cannot be combined with providers."
        raise ValueError(msg)
    return tuple(names)


def _build_evidence_searchers(
    settings: Any, snapshot_store: Any, provider_names: Sequence[str]
) -> dict[str, Any]:  # pragma: no cover - requires network providers
    """Build one AsOfSearcher per enabled provider, each with a polite client."""
    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from common.secrets import EnvSecretProvider
    from sources.searcher import build_as_of_searcher

    searchers: dict[str, Any] = {}
    for name in provider_names:
        if name == "tavily":
            from sources.providers.tavily import TavilySearchClient, tavily_config

            http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
            tavily = TavilySearchClient(
                http=http, config=tavily_config(), secrets=EnvSecretProvider()
            )
            searchers[name] = build_as_of_searcher(
                http_client=http, client=tavily, snapshot_store=snapshot_store
            )
        elif name == "gdelt":
            from sources.providers.gdelt import GdeltAsOfSearcher
            from sources.searcher import CircuitBreakerAsOfSearcher

            # A GDELT 429 is an IP cooldown: retrying deepens it, so single-
            # attempt with a 6s politeness interval; the composite skips it
            # on failure rather than stalling the whole gather. Once the
            # cooldown is on, every further call fails for a long stretch
            # (observed: hours of solid 429s), so a circuit breaker stops
            # paying the interval+failure tax after 3 straight rate limits
            # and probes again 15 minutes later.
            gdelt_http = HttpClient(
                config=HttpConfig(
                    user_agent=settings.http_user_agent or _EVIDENCE_USER_AGENT,
                    min_interval_s=6.0,
                    max_retries=1,
                )
            )
            searchers[name] = CircuitBreakerAsOfSearcher(
                GdeltAsOfSearcher(http=gdelt_http, snapshot_store=snapshot_store),
                failure_threshold=3,
                cooldown_s=900.0,
            )
        elif name == "wikipedia":
            from sources.providers.wikipedia import WikipediaAsOfSearcher

            wiki_http = HttpClient(
                config=HttpConfig(
                    user_agent=settings.http_user_agent or _EVIDENCE_USER_AGENT,
                    min_interval_s=1.0,
                )
            )
            searchers[name] = WikipediaAsOfSearcher(http=wiki_http, snapshot_store=snapshot_store)
    return searchers


def _default_forecaster(
    providers: tuple[str, ...] | None = None,
) -> Forecaster:  # pragma: no cover - requires LLM API + hosted search
    from common.composition import build_postgres_composition
    from common.llm import LLMConfig
    from core.forecast.search import FixtureAsOfSearch
    from sources.searcher import CompositeAsOfSearcher

    comp = build_postgres_composition()
    settings = comp.settings
    # Reasoning-grade config: adaptive thinking + high effort + headroom for
    # the reasoning tokens (anthropic transport only; see common/llm/config.py).
    reasoning_cfg = LLMConfig(thinking="adaptive", effort="high", max_tokens=4096)
    reasoning = comp.structured_client("opus", config=reasoning_cfg)
    provider_names = providers if providers is not None else _evidence_provider_names()
    if provider_names == ("none",):
        # The explicit no-search ablation arm: an always-empty searcher and no
        # agentic wrap (planner rounds against nothing waste LLM budget).
        return _build_forecaster(comp, reasoning, FixtureAsOfSearch(default=()))

    snapshot_store = _snapshot_store(settings.snapshot_dir)
    by_name = _build_evidence_searchers(settings, snapshot_store, provider_names)

    def _composite(names: Sequence[str]) -> Any:
        members = [by_name[n] for n in names]
        return members[0] if len(members) == 1 else CompositeAsOfSearcher(members)

    all_names = list(by_name)
    searcher: Any = _composite(all_names)
    if settings.search_rounds > 1:
        from core.forecast.agentic_search import AgenticAsOfSearcher, BedrockQueryPlannerLLM

        # GDELT's politeness interval is too slow for the iterative planner
        # loop: it contributes its one bounded query on the seed round only.
        follow_up_names = [n for n in all_names if n != "gdelt"] or all_names
        searcher = AgenticAsOfSearcher(
            inner=_composite(follow_up_names),
            planner=BedrockQueryPlannerLLM(comp.structured_client("opus")),
            max_rounds=settings.search_rounds,
            max_queries_total=settings.search_queries,
            seed_inner=_composite(all_names) if "gdelt" in all_names else None,
        )
    return _build_forecaster(comp, reasoning, searcher)


_SERIES_ESTIMATOR_ENV = "DELPHI_SERIES_ESTIMATOR"


def _default_series_estimator(settings: Any) -> Any:  # pragma: no cover - network wiring
    """Wire the deterministic series-threshold estimator (on by default).

    ``DELPHI_SERIES_ESTIMATOR=0`` disables it (e.g. an ablation arm).
    """
    import os

    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from forecaster.stages.series_estimate import SeriesEvidenceEstimator
    from sources.series import (
        DbnomicsSeriesProvider,
        FredSeriesProvider,
        SeriesRouter,
        YahooChartSeriesProvider,
    )

    if os.environ.get(_SERIES_ESTIMATOR_ENV, "1") == "0":
        return None
    http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
    router = SeriesRouter(
        {
            "fred": FredSeriesProvider(http=http),
            "yfinance": YahooChartSeriesProvider(http=http),
            "dbnomics": DbnomicsSeriesProvider(http=http),
        }
    )
    return SeriesEvidenceEstimator(source=router)


def _build_forecaster(
    comp: Any, reasoning: Any, searcher: Any
) -> Forecaster:  # pragma: no cover - requires LLM API
    import logging

    from common.llm import LLMConfig
    from core.forecast.leakage_judge import BedrockLeakageJudgeLLM, LeakageJudge
    from core.forecast.llm import BedrockForecastLLM
    from core.forecast.supervisor import BedrockSupervisorLLM
    from forecaster.calibration_artifact import load_calibration_artifact

    settings = comp.settings
    # Reasoning-grade config for the draw/supervisor clients: adaptive thinking
    # + high effort (anthropic transport only; see common/llm/config.py).
    reasoning_cfg = LLMConfig(thinking="adaptive", effort="high", max_tokens=4096)
    if settings.calibration_artifact_path:
        # A bad artifact must fail loudly here, never silently fall back (§2.5).
        calibration = load_calibration_artifact(settings.calibration_artifact_path)
    else:
        calibration = None
        logging.getLogger(__name__).warning(
            "DELPHI_CALIBRATION_ARTIFACT is not set: forecasting with the identity "
            "recalibrator and fixed alpha. Fit one with `delphi calibration fit`."
        )
    return Forecaster(
        series_estimator=_default_series_estimator(settings),
        intake=IntakeService(llm=reasoning, store=comp.registry_store),
        searcher=searcher,
        reasoning_llm=reasoning,
        forecast_llm=BedrockForecastLLM(comp.structured_client("opus", config=reasoning_cfg)),
        supervisor_llm=BedrockSupervisorLLM(comp.structured_client("fable", config=reasoning_cfg)),
        leakage_judge=LeakageJudge(BedrockLeakageJudgeLLM(comp.structured_client("opus"))),
        registry_store=comp.registry_store,
        calibration=calibration,
        aggregator=settings.aggregator,
        runs_per_agent=settings.runs_per_agent,
        evidence_subset_fraction=settings.evidence_subset_fraction,
        max_subquestion_searches=settings.subquestion_searches,
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


_EPHEMERAL_LEDGER_ENV = "DELPHI_ALLOW_EPHEMERAL_TRIALS_LEDGER"


def _eval_trials_ledger(
    *,
    pg_dsn: str | None,
    cap: int,
    env: Mapping[str, str] | None = None,
) -> BudgetLedger:
    """Build the §2.4 trials ledger for guarded evals — durable, or loudly not.

    Without Postgres there is no durable global trials count, so guarded
    evaluation REFUSES to run rather than silently degrading to an unenforced
    in-process ledger (the §2.4 anti-overfitting guarantee would be void while
    every number still printed normally). ``DELPHI_ALLOW_EPHEMERAL_TRIALS_LEDGER=1``
    is the explicit local-development opt-out; the run still warns here and the
    rendered report still carries the EPHEMERAL-ledger warning.
    """
    import os

    from core.orchestration.budget import InMemoryBudgetLedger, PostgresBudgetLedger

    if pg_dsn:
        return PostgresBudgetLedger.connect(pg_dsn, cap=cap)
    e = os.environ if env is None else env
    if e.get(_EPHEMERAL_LEDGER_ENV) == "1":
        print(
            "WARNING: DELPHI_PG_DSN is unset — using an EPHEMERAL in-memory "
            "trials ledger. Draws from this run are NOT recorded durably; the "
            "global trials count (CLAUDE.md §2.4) is not being enforced."
        )
        return InMemoryBudgetLedger(cap=cap)
    msg = (
        "No durable trials ledger: DELPHI_PG_DSN is unset, so guarded "
        "evaluations cannot draw down the global trials budget (CLAUDE.md "
        "§2.4). Set DELPHI_PG_DSN, or export "
        f"{_EPHEMERAL_LEDGER_ENV}=1 to explicitly accept an ephemeral, "
        "non-enforcing ledger for local development."
    )
    raise RuntimeError(msg)


def _default_eval_context(
    suite: str,
    *,
    calibration_artifact: str | None = None,
    no_search: bool = False,
    sample: int | None = None,
    sample_seed: int = 0,
    resume_file: str | None = None,
    exclude_fit_questions: bool = False,
    only_sources: str | None = None,
) -> EvalContext:  # pragma: no cover - network + LLM + DB
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
    from benchmarks.suites import (
        build_eval_context,
        constant_baseline,
        forecaster_fn,
        records_baseline,
        sample_records,
    )
    from common.http.client import HttpClient
    from common.http.config import HttpConfig
    from common.secrets import EnvSecretProvider
    from common.settings import load_settings
    from core.forecast.leakage_judge import BedrockLeakageJudgeLLM, LeakageJudge
    from evaluation.harness import EvalHarness

    settings = load_settings()
    http = HttpClient(config=HttpConfig(user_agent=settings.http_user_agent))
    providers = ("none",) if no_search else None
    forecast_fn = forecaster_fn(_default_forecaster(providers))

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
        if sample is not None:
            records = sample_records(records, n=sample, seed=sample_seed)
        adapter: BenchmarkAdapter = MetaculusAdapter.from_records(records)
        baselines.append(consensus_baseline(adapter, price_key="community_prediction"))
    elif suite == "forecastbench":
        from benchmarks.suites import filter_records_by_source

        question_set = os.environ["DELPHI_FORECASTBENCH_QUESTION_SET"]
        resolution_set = os.environ.get("DELPHI_FORECASTBENCH_RESOLUTION_SET")
        records = ForecastBenchFetcher(http=http).fetch(
            question_set=question_set, resolution_set=resolution_set
        )
        records = _filter_eval_records(
            records, resolved_after=resolved_after, max_questions=max_questions
        )
        if only_sources:
            records = filter_records_by_source(records, only_sources.split(","))
        if sample is not None:
            records = sample_records(records, n=sample, seed=sample_seed)
        adapter = ForecastBenchAdapter.from_records(records)
        baselines.append(records_baseline(records, source="forecastbench"))
        baselines.append(constant_baseline(records, source="forecastbench"))
    else:
        valid = ", ".join(_EVAL_SUITES)
        msg = f"unknown --suite {suite!r}; choose one of: {valid}."
        raise ValueError(msg)

    from common.composition import build_postgres_composition

    comp = build_postgres_composition()
    judge = LeakageJudge(BedrockLeakageJudgeLLM(comp.structured_client("opus")))

    ledger = _eval_trials_ledger(pg_dsn=settings.pg_dsn, cap=settings.global_trials_budget)
    calibration = None
    if calibration_artifact:
        from forecaster.calibration_artifact import load_calibration_artifact

        # A bad artifact must fail loudly, never silently fall back (§2.5).
        calibration = load_calibration_artifact(calibration_artifact)
    resume_tag = (
        f"suite={suite}|no_search={no_search}|artifact={calibration_artifact or ''}"
        f"|sources={only_sources or ''}"
    )
    return build_eval_context(
        adapter,
        forecast_fn,
        harness=EvalHarness(budget_ledger=ledger, holdout=_holdout_governor_from_env()),
        judge=judge,
        extra_baselines=tuple(baselines),
        calibration=calibration,
        resume_path=resume_file,
        resume_tag=resume_tag,
        exclude_fit_questions=exclude_fit_questions,
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
    # Every live harvest/score tick feeds the Stage-2 training corpus (§4):
    # harvest writes the pending (question, workflow, evidence, forecast) row,
    # score completes it with the resolution + proper score.
    from conductor.corpus import CorpusWriter, FileCorpusStore

    corpus_path = os.environ.get("DELPHI_CORPUS_PATH", _DEFAULT_CORPUS_PATH)
    corpus_writer = CorpusWriter(store=store, corpus=FileCorpusStore(corpus_path))
    return LiveContext(
        harvest_job=HarvestJob(conductor=_default_conductor(), corpus_writer=corpus_writer),
        score_job=ScoreJob(
            store=store,
            resolution_service=ResolutionService(
                store=store, source=BenchmarkResolutionSource(resolutions)
            ),
            corpus_writer=corpus_writer,
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

    The async job surface (POST /v1/forecast/jobs) persists jobs to Postgres
    whenever ``DELPHI_PG_DSN`` is set, so a poll landing on any gunicorn
    worker/instance sees every job; the in-memory fallback is single-process
    (local ``delphi serve`` only).
    """
    import os

    from api.jobs import InMemoryJobStore, JobManager, JobStore, PostgresJobStore
    from api.routes import ForecastService
    from api.server import forecast_runner
    from common.settings import load_settings

    token = auth_token or os.environ.get("DELPHI_SECRET_API_TOKEN") or None
    forecaster = _default_forecaster()
    conductor = HeuristicConductor(forecaster=forecaster)
    store = _default_store()
    # The API's intake surfaces (/v1/classify, /v1/formalize) preview intake
    # without recording; the store is only wired for constructor completeness.
    intake = IntakeService(llm=_default_llm(), store=store)
    service = ForecastService(
        forecaster=forecaster, conductor=conductor, store=store, intake=intake
    )
    settings = load_settings()
    job_store: JobStore = (
        PostgresJobStore.connect(settings.pg_dsn) if settings.pg_dsn else InMemoryJobStore()
    )
    jobs = JobManager(
        store=job_store,
        runner=forecast_runner(service),
        workers=settings.job_workers,
        stale_after_s=float(settings.job_stale_after_s),
    )
    return DelphiApp(service, auth_token=token, jobs=jobs)


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
    clock: Callable[[], datetime] | None = None,
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
            eval_context = _default_eval_context(  # pragma: no cover - wired suite
                args.suite,
                calibration_artifact=args.calibration_artifact,
                no_search=args.no_search,
                sample=args.sample,
                sample_seed=args.sample_seed,
                resume_file=args.resume_file,
                exclude_fit_questions=args.exclude_fit_questions,
                only_sources=args.only_sources,
            )
        return cmd_eval(args, context=eval_context)
    if args.command == "calibration" and args.calibration_command == "fit":
        return cmd_calibration_fit(
            args,
            store=store if store is not None else _default_store(),
            clock=clock,
        )
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
