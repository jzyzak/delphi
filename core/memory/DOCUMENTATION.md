# Agent Memory — Full Documentation

This document describes **every component** of the agent memory / semantic recall layer in `core/memory/`. For a one-page quick start, see [README.md](README.md). Memory builds on the experiment registry (Prompt 03, [`../registry/DOCUMENTATION.md`](../registry/DOCUMENTATION.md)) and is consumed by the research agent (Prompt 11, not yet built).

Agent memory is a **derived, rebuildable index** over the registry experiment graph. It answers: *"Have we tried something like this?"* and *"What did we learn in this niche?"* — including from **failures**. The registry remains the immutable source of truth; memory is a cache that can be dropped and reconstructed without affecting retrospective-evaluation reproducibility.

---

## Table of contents

1. [Conceptual model](#1-conceptual-model)
2. [Module map and public API](#2-module-map-and-public-api)
3. [Embedder layer (`embedder.py`)](#3-embedder-layer-embedderpy)
4. [Document assembly (`index.py`)](#4-document-assembly-indexpy)
5. [Vector index (`index.py`)](#5-vector-index-indexpy)
6. [Recall API (`recall.py`)](#6-recall-api-recallpy)
7. [PostgreSQL schema (`migrations/`)](#7-postgresql-schema-migrations)
8. [Registry integration](#8-registry-integration)
9. [Semantic recall vs fingerprint accounting](#9-semantic-recall-vs-fingerprint-accounting)
10. [Backends: in-memory and Postgres](#10-backends-in-memory-and-postgres)
11. [Error taxonomy](#11-error-taxonomy)
12. [Logging](#12-logging)
13. [Testing (M1–M7 + §8)](#13-testing-m1m7--8)
14. [Design decisions and deferred work](#14-design-decisions-and-deferred-work)
15. [End-to-end walkthroughs](#15-end-to-end-walkthroughs)
16. [Setup and operational status](#16-setup-and-operational-status)
17. [Known limitations and improvement opportunities](#17-known-limitations-and-improvement-opportunities)

---

## 1. Conceptual model

Memory has four load-bearing ideas. Every feature is a direct consequence.

### 1.1 Derived, not authoritative

Memory is an **index**, not a second source of truth. All content is assembled by **reading** registry records (`Experiment`, `Result`, `Decision`) and embedding a deterministic text representation. The index:

- **Never writes** to `registry_events`.
- **Never mutates** experiment payloads.
- **Can be fully rebuilt** via `rebuild_from_registry()` without loss of recall fidelity (given the same embedder).

If the `memory_index` table is truncated, corrupted, or stale, call `rebuild_from_registry()`. Retrospective-evaluation reproducibility depends only on the registry and harness — never on the embedding index.

### 1.2 Failures are first-class

Rejected and abandoned experiments are valuable information. Memory indexes them identically to promoted ones and exposes them through the same recall filters (`outcome="rejected"`, `outcome="abandoned"`, or `outcome="any"`). An agent assembling context for a new hypothesis in `us_elections` should see prior failures as readily as successes.

### 1.3 Semantic ≠ accounting

Two distinct mechanisms serve different purposes:

| Mechanism | What it measures | Authoritative? |
|-----------|------------------|----------------|
| `trial_fingerprint` (registry) | Exact trial identity: `(spec_hash, params, data_snapshot, universe_spec)` | **Yes** — honest trial counting (Prompts 03/06) |
| `near_duplicates()` (memory) | Semantic similarity of a candidate spec to prior experiments | **No — advisory only** |

`near_duplicates()` warns an agent *"this looks very close to trial X"* to save budget and prompt better hypotheses. It **never** decides what counts as a trial. Counting remains `registry.duplicate_experiment_ids(fingerprint)`.

### 1.4 Context for the agent

The recall surface exists to assemble relevant prior hypotheses, lessons, and near-duplicates into the research agent's prompt context (Prompt 11). Memory does not generate strategies, run evaluations, or write experiments.

### What this prevents

| Anti-pattern | How memory prevents or mitigates it |
|--------------|-------------------------------------|
| Redundant research on near-identical hypotheses | Semantic recall + advisory `near_duplicates()` surface prior work |
| Losing the "what doesn't work" map | Failures indexed and recall-filterable by outcome |
| Memory becoming a second source of truth | Full rebuild from registry; no write path to registry |
| Semantic similarity substituting for trial accounting | `near_duplicates()` explicitly documented and tested as advisory |
| Non-reproducible recall | Deterministic embedder; fixed assembly rules; no network in tests |

---

## 2. Module map and public API

```
core/memory/
├── __init__.py          # Public re-exports
├── embedder.py          # Embedder Protocol + DeterministicEmbedder
├── index.py             # Document assembly, VectorIndex ABC, backends
├── recall.py            # MemoryRecall: recall, lessons, near_duplicates
├── migrations/
│   └── 0001_init.sql    # memory_index table + pgvector extension
├── README.md            # One-page quick start
└── DOCUMENTATION.md     # This file
```

### Public exports (`core/memory/__init__.py`)

| Symbol | Module | Role |
|--------|--------|------|
| `Embedder` | `embedder` | Protocol for mockable text→vector mapping |
| `DeterministicEmbedder` | `embedder` | Default offline concrete embedder |
| `VectorIndex` | `index` | ABC for rebuildable vector indexes |
| `InMemoryVectorIndex` | `index` | In-process numpy cosine backend |
| `PostgresVectorIndex` | `index` | PostgreSQL + pgvector backend |
| `Recollection` | `index` | One recalled experiment + score |
| `IndexDocument` | `index` | Assembled index payload (pre-embedding) |
| `RecallOutcome` | `index` | Filter literal: `promoted` / `rejected` / `abandoned` / `any` |
| `MemoryRecall` | `recall` | High-level recall API |
| `assemble_document` | `index` | Fold registry records → embeddable text |
| `render_spec_description` | `index` | Deterministic spec text from `ReproMetadata` |
| `index_experiment` | `index` | Index one experiment by id |
| `MemoryError` | `index` | Base exception |
| `DimensionMismatchError` | `index` | Embedding dimension mismatch |

---

## 3. Embedder layer (`embedder.py`)

### 3.1 `Embedder` Protocol

```python
@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

**Contract:**

- **Batch interface** — `embed()` accepts a sequence of strings and returns one vector per string, in order.
- **Fixed dimension** — `dim` is constant for the lifetime of the embedder instance.
- **L2-normalized output** — each returned vector should be a unit vector (or zero vector for empty input).
- **Mockable** — structural typing via `Protocol`; tests inject deterministic fixtures without subclassing.
- **Pure** — no network, no wall-clock reads, no hidden global state in the default implementation.

Production may later swap in a model-backed embedder (e.g. Bedrock Titan, Anthropic) behind the same protocol. Tests **never** call a live model.

### 3.2 `DeterministicEmbedder`

The default concrete implementation. Hash-based token n-gram embedding:

1. **Tokenize** — lowercase alphanumeric tokens via `[a-z0-9]+`.
2. **Unigram buckets** — each token hashes (SHA-256 → first 4 bytes mod `dim`) into a bucket; bucket count += 1.
3. **Bigram buckets** — consecutive token pairs also increment a bucket.
4. **Normalize** — L2-normalize the count vector to a unit vector.

**Properties:**

| Property | Value |
|----------|-------|
| Default `dim` | 128 (must be ≥ 8) |
| Determinism | Identical text → identical vector, always |
| Network | None |
| Seeds | None required (hash is deterministic) |

**Failure modes:**

- `dim < 8` → `ValueError("Embedding dimension must be at least 8.")`
- Empty `texts` sequence → returns `[]` (no vectors)

**Note on Postgres dimension:** The migration hardcodes `vector(128)`. If you use a different `DeterministicEmbedder(dim=...)`, the Postgres backend requires a matching migration or the same default dimension.

---

## 4. Document assembly (`index.py`)

Before embedding, registry records are folded into an `IndexDocument` and a multi-line `embedded_text` string.

### 4.1 `assemble_document(store, experiment) -> IndexDocument`

**Read-only** over `store`. Steps:

1. Fetch `decisions_for(experiment_id)` and `results_for(experiment_id)`.
2. Take the **latest** decision (last in append order) and **latest** result.
3. Derive `outcome`, `lessons`, `spec_description`, and `embedded_text`.
4. Return frozen `IndexDocument`.

**Raises:** `RecordNotFoundError` if `experiment.experiment_id` does not exist in the store (via `decisions_for` / `results_for` precondition checks).

### 4.2 Outcome mapping

Registry `Decision.outcome` uses verb forms; the index uses past-participle filter labels:

| Registry `DecisionOutcome` | Index `IndexOutcome` |
|----------------------------|----------------------|
| `promote` | `promoted` |
| `reject` | `rejected` |
| `abandon` | `abandoned` |
| *(no decision yet)* | `pending` |

**Latest decision wins.** If an experiment has multiple decisions (corrections are append-only), only the last decision's outcome is indexed. This mirrors `experiments_by_outcome()` in the registry.

### 4.3 Lessons derivation

The registry has **no native `lessons` field**. Memory derives lessons at index time from:

1. `Decision.rationale` (if non-empty after strip)
2. `Decision.evidence` (canonical JSON)
3. `Result.metrics` (canonical JSON, from latest result)

Parts are joined with `" | "`. Example:

```
Base rate dominated the inside view. | evidence={"gate":"calibration"} | metrics={"brier":0.21}
```

If a future registry record kind stores explicit lessons, assembly can be updated to prefer that field; today this derivation is the contract.

### 4.4 Spec description

The registry stores `repro.spec_kind`, `repro.spec_hash`, and `repro.params` — not human-readable spec text. `render_spec_description()` produces a deterministic string:

```
dsl spec (hash=spec-hash-001), params={"lookback":20,"threshold":1.5}
```

`params` are serialized via `registry.fingerprint.canonical_json` (sorted keys, tight JSON) so identical logical params always produce identical text.

### 4.5 Embedded text format

The string passed to the embedder is newline-separated:

```
Hypothesis: <experiment.hypothesis>
Economic rationale: <experiment.economic_rationale>
Spec: <render_spec_description(repro)>
Outcome: <promoted|rejected|abandoned|pending>
Lessons: <derived lessons>    # omitted if lessons string is empty
```

**No secrets:** All source fields are registry text already scanned at write time (`SecretInRecordError` in `core/registry/store.py`). Memory does not add credentials or env secrets to embedded text.

### 4.6 `IndexDocument` and `Recollection`

**`IndexDocument`** (dataclass, frozen) — internal pre-embedding payload:

| Field | Type | Source |
|-------|------|--------|
| `experiment_id` | `str` | `Experiment.experiment_id` |
| `niche` | `str` | `Experiment.niche` |
| `outcome` | `IndexOutcome` | Latest decision or `pending` |
| `trial_fingerprint` | `str` | `Experiment.trial_fingerprint` |
| `embedded_text` | `str` | Assembled (§4.5) |
| `lessons` | `str` | Derived (§4.3) |
| `knowledge_time` | `datetime` | `Experiment.knowledge_time` |

**`Recollection`** (Pydantic, frozen) — one search hit returned to callers:

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | `str` | Registry experiment id |
| `niche` | `str` | Experiment niche |
| `outcome` | `IndexOutcome` | Indexed outcome |
| `score` | `float` | Cosine similarity in `[-1, 1]` |
| `embedded_text` | `str` | Full indexed text (for context injection) |
| `lessons` | `str` | Derived lessons string |
| `trial_fingerprint` | `str` | Exact trial identity (for cross-reference with registry) |

---

## 5. Vector index (`index.py`)

### 5.1 `VectorIndex` ABC

Shared logic lives on the ABC; subclasses implement storage primitives only (mirrors `RegistryStore` design).

**Constructor:** `VectorIndex(store: RegistryStore, embedder: Embedder)`

Holds references to the canonical registry (read-only) and the embedder.

#### `index(experiment: Experiment) -> None`

Incremental add/update for one experiment:

1. `assemble_document(store, experiment)`
2. `embedder.embed([embedded_text])` → one vector
3. `_upsert(doc, vector)`
4. Log `memory_indexed` event

Re-indexing the same `experiment_id` overwrites the prior entry (upsert semantics).

#### `rebuild_from_registry() -> int`

Full reconstruction:

1. `_clear()` — drop all index entries
2. `store.all_experiments()` — enumerate every experiment (read-only registry API added for this purpose)
3. `index(exp)` for each experiment in knowledge-time order
4. Log `memory_rebuild` with count
5. Return number of experiments indexed

**Contract (M3/M7):** Given the same registry state and embedder, `rebuild_from_registry()` produces **identical recall** to incremental `index()` on the same records.

#### `search(vector, *, niche=None, outcome="any", k=10) -> list[Recollection]`

Low-level nearest-neighbor search:

- **Similarity:** cosine similarity (higher = more similar)
- **Filters:** optional `niche` exact match; `outcome` filter (`promoted` / `rejected` / `abandoned` / `any`)
- **Ranking:** descending score; ties broken by `experiment_id` (in-memory) or database order (Postgres)
- **Limit:** top `k` results (`k=0` returns empty list)

**Validation:**

- Invalid `outcome` → `ValueError`
- `k < 0` → `ValueError`
- Vector dimension mismatch with stored embeddings → `DimensionMismatchError`

#### Abstract primitives (subclass implements)

| Method | Role |
|--------|------|
| `_upsert(doc, vector)` | Insert or replace one index entry |
| `_clear()` | Remove all entries |
| `_search_vectors(vector, niche, outcome, k)` | Filtered nearest-neighbor query |

#### `index_experiment(store, index, experiment_id)`

Convenience: `get_experiment(id)` then `index(exp)`. Raises `RecordNotFoundError` if missing.

---

## 6. Recall API (`recall.py`)

### 6.1 `MemoryRecall`

High-level facade for agent context assembly.

**Constructor:**

```python
MemoryRecall(embedder: Embedder, index: VectorIndex, store: RegistryStore)
```

Holds embedder + index + store. Recall methods embed queries and delegate to `index.search()`. The `store` reference is retained for future extensions (e.g. hydrating full experiment records); current recall paths are read-only over the index.

### 6.2 `recall(*, query, niche=None, outcome="any", k=10) -> list[Recollection]`

**Primary semantic recall entry point.**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural-language or keyword query (non-empty) |
| `niche` | `str \| None` | `None` | If set, only experiments in this niche |
| `outcome` | `RecallOutcome` | `"any"` | Filter by indexed outcome |
| `k` | `int` | `10` | Maximum results |

**Behavior:**

1. Validate `query` is non-empty and `k >= 0`
2. `embedder.embed([query])` → query vector
3. `index.search(vector, niche=..., outcome=..., k=...)`
4. Return ranked `Recollection` list

**Contract:** Deterministic given a fixed embedder and index state. Never mutates registry or index (read-only search).

### 6.3 `lessons(*, query, niche=None, k=10) -> list[str]`

Distilled lessons for prompt injection.

1. Calls `recall(query=..., niche=..., outcome="any", k=...)`
2. Returns `[r.lessons for r in recollections if r.lessons]` — preserves recall ranking, drops empty lessons

Use when the agent needs prior learnings without full `Recollection` payloads.

### 6.4 `near_duplicates(*, spec_description, threshold) -> list[Recollection]`

**ADVISORY** semantic near-duplicate detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `spec_description` | `str` | Candidate spec text (non-empty); typically from `render_spec_description()` plus hypothesis keywords |
| `threshold` | `float` | Minimum cosine similarity in `[0.0, 1.0]` |

**Behavior:**

1. Embed `spec_description`
2. Search top 100 candidates (`outcome="any"`, no niche filter)
3. Return those with `score >= threshold`

**Critical boundary:** This is **not** trial accounting. Two experiments with different `trial_fingerprint` values can still be flagged as near-duplicates if their embedded text is semantically similar. Conversely, identical fingerprints are detected authoritatively by `registry.duplicate_experiment_ids()`, not by memory.

**Validation:**

- Empty `spec_description` → `ValueError`
- `threshold` outside `[0, 1]` → `ValueError`

---

## 7. PostgreSQL schema (`migrations/`)

### 7.1 Migration strategy

Same pattern as registry and PIT:

- Raw SQL files in `core/memory/migrations/`, applied in sorted glob order
- `PostgresVectorIndex.apply_migrations()` runs on `connect(..., migrate=True)`
- Idempotent DDL (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`)
- No migration tracking table — safe to re-run

### 7.2 `0001_init.sql`

**Extension:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Requires PostgreSQL with pgvector installed.

**Table `memory_index`:**

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | `TEXT PRIMARY KEY` | Registry experiment id |
| `niche` | `TEXT NOT NULL` | Denormalized filter column |
| `outcome` | `TEXT NOT NULL` | `promoted` / `rejected` / `abandoned` / `pending` |
| `trial_fingerprint` | `TEXT NOT NULL` | Denormalized for advisory cross-reference |
| `embedded_text` | `TEXT NOT NULL` | Full text that was embedded |
| `lessons` | `TEXT NOT NULL DEFAULT ''` | Derived lessons |
| `embedding` | `vector(128) NOT NULL` | pgvector embedding |
| `knowledge_time` | `TIMESTAMPTZ NOT NULL` | From experiment record |
| `updated_at` | `TIMESTAMPTZ NOT NULL` | Last index write time |

**Indexes:**

| Index | Purpose |
|-------|---------|
| `memory_index_niche_idx` | Niche filter |
| `memory_index_outcome_idx` | Outcome filter |
| `memory_index_embedding_idx` | IVFFlat cosine (`vector_cosine_ops`, `lists=100`) |

**Upsert:** `ON CONFLICT (experiment_id) DO UPDATE` — re-indexing updates all columns.

**Clear:** `TRUNCATE memory_index` during `rebuild_from_registry()`.

### 7.3 Search query (Postgres)

```sql
SELECT ..., 1 - (embedding <=> %(embedding)s) AS score
FROM memory_index
WHERE <niche/outcome filters>
ORDER BY embedding <=> %(embedding)s
LIMIT %(k)s
```

`<=>` is cosine distance under `vector_cosine_ops`. For L2-normalized vectors, `score = 1 - distance` equals cosine similarity.

---

## 8. Registry integration

### 8.1 Read path

Memory **only reads** the registry via `RegistryStore` public methods:

| Method | Used for |
|--------|----------|
| `all_experiments()` | Full rebuild enumeration |
| `get_experiment(id)` | Single experiment fetch |
| `decisions_for(id)` | Latest outcome + lessons |
| `results_for(id)` | Latest metrics for lessons |

### 8.2 `all_experiments()` (registry addition)

Added to `RegistryStore` in `core/registry/store.py` for rebuild support:

```python
def all_experiments(self) -> tuple[Experiment, ...]:
    return self._experiments_where(lambda _ev: True)
```

Returns all experiments sorted by `(knowledge_time, experiment_id)`. Purely additive, read-only — does not weaken registry integrity.

### 8.3 Write path

Memory has **no write path** to the registry. Recording experiments, results, and decisions remains the registry's job (and later the research agent orchestration). After registry writes, callers invoke `index.index(experiment)` or schedule a rebuild.

### 8.4 Typical indexing workflow

```
1. store.record_experiment(...)     # registry write
2. store.record_result(...)         # optional
3. store.record_decision(...)       # optional but needed for outcome/lessons
4. index.index(store.get_experiment(exp_id))   # memory index update
```

Experiments without decisions are indexed with `outcome="pending"` and lessons from results only (if any).

---

## 9. Semantic recall vs fingerprint accounting

```
                    ┌─────────────────────────────────────┐
                    │         Candidate proposal          │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
   trial_fingerprint()      near_duplicates()          recall()
   (registry/fingerprint)   (memory, advisory)        (memory, context)
              │                       │                       │
              ▼                       ▼                       ▼
   Exact match? Debit         Similar prior work?      Ranked prior
   trials ledger.             Warn agent only.         experiments +
   AUTHORITATIVE.             NOT authoritative.       lessons.
```

**When to use which:**

| Question | Use |
|----------|-----|
| "Is this the exact same trial as before?" | `trial_fingerprint` + `duplicate_experiment_ids()` |
| "Have we explored something semantically similar?" | `near_duplicates()` or `recall()` |
| "What did we learn in this niche?" | `recall()` + `lessons()` |

---

## 10. Backends: in-memory and Postgres

### 10.1 `InMemoryVectorIndex`

| Aspect | Detail |
|--------|--------|
| Storage | `dict[str, tuple[IndexDocument, list[float]]]` keyed by `experiment_id` |
| Search | Brute-force cosine over all entries passing filters |
| Use case | Unit tests (M1–M7), local dev, CI without Postgres |
| Dependencies | `numpy` only |

**Tie-breaking:** `(-score, experiment_id)` sort key.

### 10.2 `PostgresVectorIndex`

| Aspect | Detail |
|--------|--------|
| Connection | `psycopg` v3 sync; `register_vector(conn)` for pgvector type |
| Factory | `PostgresVectorIndex.connect(dsn, store, embedder, migrate=True, clock=...)` |
| DSN convention | `DELPHI_PG_DSN` environment variable (same as registry/PIT) |
| Lifecycle | Context manager + `close()` |
| Clock injection | `clock: Callable[[], datetime]` for deterministic `updated_at` in tests |

**Note:** `connect()` opens a **separate** connection from `PostgresRegistryStore.connect()` even when given the same DSN. Callers typically hold both handles and pass the same `RegistryStore` instance to the index constructor.

### 10.3 Backend selection guide

| Environment | Backend |
|-------------|---------|
| `pytest tests/memory` (default) | `InMemoryVectorIndex` |
| `pytest -m postgres` | `PostgresVectorIndex` |
| Production agent service | `PostgresVectorIndex` |

Both backends share identical assembly, embedding, and search semantics via the ABC.

---

## 11. Error taxonomy

| Exception | Module | When raised |
|-----------|--------|-------------|
| `MemoryError` | `index` | Base class; embedder returned no vectors |
| `DimensionMismatchError` | `index` | Query vector dim ≠ stored/embedder dim |
| `ValueError` | `index`, `recall`, `embedder` | Invalid `k`, `outcome`, `query`, `threshold`, or `dim` |
| `RecordNotFoundError` | `registry` (propagated) | `index_experiment()` or `assemble_document()` with missing experiment |

Memory does not define separate exceptions for empty recall results — callers receive `[]`.

---

## 12. Logging

Uses `structlog` with module-level `_LOG = structlog.get_logger(__name__)`.

| Event | When | Fields |
|-------|------|--------|
| `memory_indexed` | After `index()` | `experiment_id`, `niche`, `outcome` |
| `memory_rebuild` | After `rebuild_from_registry()` | `count` |

No bare `print` in library code.

---

## 13. Testing (M1–M7 + §8)

Tests live in `tests/memory/`. All component tests use a **`ClusteredFixtureEmbedder`** that maps one topic's vocabulary to one unit vector and a second topic's to another — making semantic ranking testable without a real model.

### Mandatory component tests

| ID | Requirement | Test class / method |
|----|-------------|---------------------|
| **M1** | Semantic recall ranks relevant above irrelevant | `TestSemanticRecall` |
| **M2** | Rejected/abandoned retrievable by niche/outcome | `TestFailuresFirstClass` |
| **M3** | `rebuild_from_registry()` reproduces incremental recall | `TestRebuildableIndex::test_rebuild_matches_incremental_recall` |
| **M4** | Near-duplicate flags semantic match, decoupled from fingerprint | `TestNearDuplicateAdvisory` |
| **M5** | Fixed embedder → deterministic recall | `TestDeterminism` |
| **M6** | No secrets in embedded text; embedder mockable | `TestNoSecretsMockable` |
| **M7** | New experiment recallable after incremental index; reindex reproduces state | `TestRebuildableIndex::test_new_experiment_recallable_after_incremental_index` |

### §8 unit coverage (happy / edge / failure)

| Area | Cases |
|------|-------|
| `DeterministicEmbedder` | Same text → same vector; `dim < 8` rejected; empty batch |
| `assemble_document` | Pending without decision; missing experiment raises |
| `MemoryRecall` validation | Empty query; invalid threshold; negative `k` |
| `VectorIndex.search` | Dimension mismatch with indexed data |
| `PostgresVectorIndex` | Mocked migration execution |
| Postgres integration | Round-trip recall + rebuild parity (`@pytest.mark.postgres`, skipped without `DELPHI_PG_DSN`) |

### Running tests

```bash
# Offline (default) — 20 passed, 2 postgres skipped
uv run pytest tests/memory

# With real Postgres + pgvector
DELPHI_PG_DSN=postgresql://user:pass@localhost:5432/delphi \
  uv run pytest tests/memory -m postgres
```

---

## 14. Design decisions and deferred work

### Decisions made

| Decision | Rationale | Reversible? |
|----------|-----------|-------------|
| Hash-based `DeterministicEmbedder` as default | Offline, reproducible, no AWS/network in tests | Yes — swap via `Embedder` Protocol |
| Dual backend (InMemory + Postgres) | M1–M7 run without database; production uses pgvector | Yes |
| Lessons derived from decision/result | Registry has no `lessons` field today | Yes — prefer native field if added |
| Spec description from hash + params | Registry stores no human spec text | Yes — link to DSL serializer later |
| Separate `memory_index` table | Never touch `registry_events`; clear derived/cache boundary | Harder to reverse (schema migration) |
| `vector(128)` in DDL | Matches default `DeterministicEmbedder.dim` | Requires migration to change |
| IVFFlat index | Fast approximate search at scale | Can switch to HNSW or sequential scan |
| `near_duplicates` scans top 100 | Practical cap for advisory signal | Tunable constant |

### Deferred (out of scope for Prompt 10)

| Item | Prompt |
|------|--------|
| Research agent / strategy generation | 11 |
| Writing experiments to registry | 03 (callers) |
| Trials ledger accounting math | 06 |
| Model-backed embedder (Bedrock/Anthropic) | Future |
| Hydrating full `Experiment` objects in recall results | 11 |
| Automatic index-on-append hook in registry | Orchestration layer |
| `DOCUMENTATION.md`-level DSL spec text resolution | DSL / registry linkage |

---

## 15. End-to-end walkthroughs

### 15.1 Incremental indexing after a failed experiment

```python
from core.registry.models import DecisionInput, ResultInput
from core.registry.store import InMemoryRegistryStore
from core.memory import (
    DeterministicEmbedder,
    InMemoryVectorIndex,
    MemoryRecall,
)
from tests.registry.conftest import make_experiment_input  # example only

store = InMemoryRegistryStore()
embedder = DeterministicEmbedder()
index = InMemoryVectorIndex(store, embedder)
recall = MemoryRecall(embedder, index, store)

# 1. Record experiment + outcome in registry
exp = make_experiment_input(
    hypothesis="Polling averages underestimate incumbent turnout.",
    niche="us_elections",
)
exp_id = store.record_experiment(exp)
store.record_result(ResultInput(experiment_id=exp_id, status="success", metrics={"brier": 0.31}))
store.record_decision(DecisionInput(
    experiment_id=exp_id,
    outcome="reject",
    deciding_component="harness.gates",
    component_version="1.0.0",
    rationale="Brier score worse than the base-rate baseline.",
))

# 2. Index in memory layer
index.index(store.get_experiment(exp_id))

# 3. Recall prior failures in the same niche
failures = recall.recall(
    query="US election polling turnout models",
    niche="us_elections",
    outcome="rejected",
    k=5,
)

# 4. Inject lessons into agent context
lessons = recall.lessons(query="US election polling", niche="us_elections", k=3)
```

### 15.2 Full rebuild after index loss

```python
# Postgres production path
registry = PostgresRegistryStore.connect(dsn, migrate=True)
embedder = DeterministicEmbedder(dim=128)
index = PostgresVectorIndex.connect(dsn, registry, embedder, migrate=True)

count = index.rebuild_from_registry()
# count == number of experiments in registry; recall now matches full history
```

### 15.3 Advisory near-duplicate check before submitting a trial

```python
from core.memory import render_spec_description
from core.registry.fingerprint import trial_fingerprint

candidate_repro = make_repro(spec_hash="spec-hash-new", params={"lookback": 25})
spec_text = render_spec_description(candidate_repro)

# Authoritative: exact fingerprint check (registry)
fingerprint = trial_fingerprint(candidate_repro)
prior_ids = store.duplicate_experiment_ids(fingerprint)  # exact duplicates

# Advisory: semantic similarity (memory)
similar = recall.near_duplicates(
    spec_description=spec_text + " US election polling drift",
    threshold=0.85,
)
# similar may include experiments with DIFFERENT fingerprints — warn only
```

### 15.4 Postgres setup for integration tests

```bash
# Requires PostgreSQL 16+ with pgvector extension
export DELPHI_PG_DSN=postgresql://user:pass@localhost:5432/delphi
uv run pytest tests/memory -m postgres -v
```

The postgres fixture truncates `registry_events` and `memory_index` before each test.

---

## 16. Setup and operational status

**Bottom line: the memory layer is library-complete for Prompt 10.** In-memory recall works with zero external services. Production-quality semantic recall over durable storage requires Postgres **with the pgvector extension** — a stricter bar than the registry alone.

### 16.1 Already done (no action needed)

| Item | Status |
|------|--------|
| Runtime dependencies (`numpy`, `pgvector`, `psycopg[binary]`, `pydantic`, `structlog`) | Declared in `pyproject.toml`; installed by `uv sync` |
| Package wiring | `core` is in hatch build targets; `core/memory` is in `pyright` `include` and `coverage` `source` |
| Schema | `migrations/0001_init.sql`; auto-applied by `PostgresVectorIndex.connect(dsn, migrate=True)` |
| CI | CI runs ruff, pyright, and pytest — **20 in-memory memory tests** run on every push |
| In-memory backend | Works out of the box; M1–M7 + §8 pass without Postgres |
| Registry enumeration | `RegistryStore.all_experiments()` supports full rebuild |

### 16.2 To use the in-memory backend (local dev, tests, agents)

Nothing beyond `uv sync`. This is the right default for unit tests, agent acceptance tests, and offline research:

```python
from core.registry.store import InMemoryRegistryStore
from core.memory import DeterministicEmbedder, InMemoryVectorIndex, MemoryRecall

store = InMemoryRegistryStore()
embedder = DeterministicEmbedder()
index = InMemoryVectorIndex(store, embedder)
recall = MemoryRecall(embedder, index, store)
```

**Caveat:** `DeterministicEmbedder` is a hash-based stand-in, not a semantic model. Ranking quality is sufficient for deterministic tests but not for production agent recall.

### 16.3 To use the Postgres + pgvector backend

Unlike the registry (core Postgres only), the memory index **requires pgvector**:

1. **PostgreSQL 16+ with pgvector installed.** A plain `postgres:16` Docker image is **not enough** unless you install the extension manually. Prefer an image that bundles pgvector, e.g. `pgvector/pgvector:pg16`, or install `postgresql-16-pgvector` on your host.
2. **Create a database** and set `DELPHI_PG_DSN` (same convention as registry, PIT, trials, orchestration).
3. **Connect registry and memory** (two connections today — see §17):

```python
from core.registry import PostgresRegistryStore
from core.memory import DeterministicEmbedder, PostgresVectorIndex, MemoryRecall

dsn = "postgresql://user:pass@host:5432/delphi"
registry = PostgresRegistryStore.connect(dsn, migrate=True)
embedder = DeterministicEmbedder(dim=128)  # must match vector(128) in migration
index = PostgresVectorIndex.connect(dsn, registry, embedder, migrate=True)
recall = MemoryRecall(embedder, index, registry)

# Initial population or recovery after index loss:
index.rebuild_from_registry()
```

4. **Verify integration tests locally:**

```bash
docker run -d --name delphi-pgvector \
  -e POSTGRES_PASSWORD=delphi -e POSTGRES_DB=delphi \
  -p 5432:5432 pgvector/pgvector:pg16

DELPHI_PG_DSN=postgresql://postgres:delphi@localhost:5432/delphi \
  uv run pytest tests/memory -m postgres -v
```

### 16.4 Open setup items (to "finish" production deployment)

These are gaps relative to the broader vision in `CLAUDE.md`, not defects in the Prompt 10 deliverable:

| Priority | Item | Why it matters | Suggested action |
|----------|------|----------------|------------------|
| **P0** | **Postgres + pgvector in CI** | CI has no database service; 2 memory Postgres tests and all other `-m postgres` suites skip in CI. The pgvector DDL path is never validated on push. | Add `services: postgres` with a pgvector image; set `DELPHI_PG_DSN`; run `pytest -m postgres`. |
| **P1** | **Model-backed embedder** | `DeterministicEmbedder` is offline/test-only. Production recall quality depends on real embeddings (Bedrock Titan, Anthropic, etc.). | Implement `BedrockEmbedder` / `AnthropicEmbedder` behind `Embedder` Protocol; pin model IDs from docs; batch API for bulk reindex. |
| **P1** | **Orchestration index hook** | The research agent calls `index_experiment()` after each gate run, but nothing schedules a **full rebuild** on deploy, embedder change, or disaster recovery. | Add an orchestration step (or `delphi memory rebuild` CLI) that runs `rebuild_from_registry()` on startup / nightly. |
| **P2** | **Shared connection / config** | DSN is passed as raw strings; registry and memory open **separate** `psycopg` connections. | Centralize DSN in a typed settings object; optional `psycopg_pool` shared across stores. |
| **P2** | **Migration version ledger** | `apply_migrations()` re-runs all `*.sql` on each connect (idempotent today, same pattern as registry). | Add `schema_migrations` table before a non-idempotent `0002_*.sql` lands. |
| **P2** | **`delphi` CLI surface** | No command to rebuild, inspect index stats, or verify recall. | e.g. `delphi memory rebuild`, `delphi memory stats` once `common/cli.py` exists. |
| **P3** | **Embedding version metadata** | Changing embedder model or `dim` silently invalidates semantic comparability across index entries. | Add `embedder_id` / `embedder_version` column; force full rebuild when version changes. |
| **P3** | **Human-readable spec text in embeddings** | Embedded spec is `hash + params` only; DSL source text is not resolved. | Resolve `spec_hash` → serialized DSL at index time (optional enrichment). |
| **P3** | **Native registry `lessons` field** | Lessons are derived proxies from decision rationale/evidence. | Add optional append-only lessons record to registry; prefer over derivation when present. |

### 16.5 Downstream wiring checklist

For memory to be **operationally live** in a running DELPHI stack:

- [x] Registry records experiments with hypothesis, rationale, niche, repro metadata
- [x] Memory indexes from registry (`index()` / `rebuild_from_registry()`)
- [x] Research agent assembles context via `MemoryRecall` (Prompt 11)
- [x] Agent calls `index_experiment()` after successful gate submission
- [ ] Orchestrator invokes agent loop on a schedule (Prompt 16) — memory is passive until indexing runs
- [ ] Production Postgres with pgvector provisioned (infra/Terraform)
- [ ] Production embedder swapped in (not `DeterministicEmbedder`)
- [ ] CI validates Postgres + pgvector path
- [ ] Runbook for index rebuild after embedder upgrade or corruption

---

## 17. Known limitations and improvement opportunities

Honest accounting of where the current implementation trades simplicity for headroom. None of these are correctness bugs relative to Prompt 10; they are quality, scale, and hardening opportunities.

### 17.1 Recall quality

| # | Limitation | Improvement |
|---|------------|-------------|
| 1 | **`DeterministicEmbedder` is not semantic.** Hash buckets over tokens approximate similarity for tests but do not capture meaning. Topic clustering in tests is fixture-driven, not emergent. | Ship a model-backed `Embedder`; keep deterministic impl for CI only. Document embedder choice in index metadata. |
| 2 | **Single embedding per experiment.** Hypothesis, rationale, spec, and lessons are concatenated into one vector. A query focused on spec similarity may be diluted by hypothesis text. | Multi-vector index (separate embeddings per field) or weighted fusion at search time. |
| 3 | **Spec description is hash + params, not DSL text.** Agents see `dsl spec (hash=…), params={…}` rather than the actual signal expression. | Resolve `spec_hash` via DSL serializer; optionally embed human-readable expression separately. |
| 4 | **Lessons are derived, not authored.** Concatenated rationale/evidence/metrics can be noisy or long for prompt injection. | Registry-native lessons field; LLM-distilled lesson summary at index time (stored in derived table only). |
| 5 | **`near_duplicates()` scans fixed top-100.** May miss similar experiments ranked below 100; threshold tuning is manual. | Expose `k` parameter; consider two-stage retrieval (broad recall → threshold filter). |

### 17.2 Scale and performance

| # | Limitation | Improvement |
|---|------------|-------------|
| 6 | **In-memory search is O(n) brute force.** Fine for tests and small indexes; does not scale to thousands of experiments. | Use Postgres backend in production; keep in-memory for tests only. |
| 7 | **IVFFlat index with `lists=100`.** Suboptimal on small corpora; requires enough rows to be effective; no `REINDEX` after bulk load. | Switch to HNSW (pgvector ≥ 0.5) or run `REINDEX` after large rebuild; tune `lists` to `sqrt(n)`. |
| 8 | **`rebuild_from_registry()` is synchronous and unbounded.** Loads and embeds every experiment in one process call. | Batch embedding API; chunked rebuild with progress logging; background job on AWS Batch. |
| 9 | **`all_experiments()` loads full experiment set.** Registry query filters in Python today (see registry §20). Rebuild cost grows with registry size. | Push enumeration to SQL; incremental rebuild (index only experiments newer than `max(knowledge_time)` in `memory_index`). |
| 10 | **Hardcoded `vector(128)` in DDL.** Changing `DeterministicEmbedder(dim=…)` without a migration breaks Postgres upsert. | Parameterize dimension in migration or add `0002` migration; store `dim` in a metadata table. |

### 17.3 Operations and consistency

| # | Limitation | Improvement |
|---|------------|-------------|
| 11 | **Index can drift from registry.** If `index_experiment()` fails after a successful gate run, recall omits the new experiment until the next rebuild. | Orchestration retry on index failure; alert on index/registry count mismatch. |
| 12 | **No staleness detection.** Callers cannot tell when an entry was last updated vs registry's latest decision. | Compare `memory_index.updated_at` to latest decision `knowledge_time`; expose `index_stale(experiment_id)`. |
| 13 | **Separate DB connections for registry and index.** Two pools, two migration runners, no shared transaction. | Shared connection factory; optional same-transaction "read registry → write index" for consistency snapshots. |
| 14 | **No embedder/version in index schema.** Rebuilding after embedder swap mixes incompatible vectors until full truncate. | `embedder_version` column; reject search if query embedder ≠ stored version. |
| 15 | **Pending experiments indexed without decisions.** Experiments recorded but not yet gated appear as `outcome="pending"`. | Optionally defer indexing until first `Decision` exists; or separate "draft" filter in recall. |

### 17.4 API and ergonomics

| # | Limitation | Improvement |
|---|------------|-------------|
| 16 | **`MemoryRecall` holds `store` but rarely uses it.** Recall returns `Recollection` summaries, not full `Experiment` / `Decision` objects. | `recall.hydrate(recollection) -> ExperimentBundle` for agent prompt assembly. |
| 17 | **No pagination on recall.** Large `k` returns unbounded payloads (`embedded_text` can be long). | Return slim hits (id, score, lessons) with optional `include_embedded_text`. |
| 18 | **Outcome filter uses index labels, not registry verbs.** Callers must know `rejected` not `reject`. | Accept both forms in API; document mapping (already in §4.2). |
| 19 | **No niche autocomplete / facet counts.** Agent cannot ask "how many failures in this niche?" without separate registry queries. | `index.facets()` returning niche × outcome counts from denormalized columns. |
| 20 | **Coverage floor may not include all memory branches in CI.** `core/memory` is in coverage source; Postgres paths only run locally. | Postgres CI job + coverage merge; or mocked pgvector tests for SQL shape. |

### 17.5 Security and integrity

| # | Limitation | Improvement |
|---|------------|-------------|
| 21 | **Embedded text trusts registry secret scan.** Memory adds no second scan; a future registry bypass would flow into embeddings. | Optional pre-embed scan reusing `_SECRET_MARKERS` from `registry/store.py`. |
| 22 | **Near-duplicate is advisory but easy to misuse.** A hurried integrator might treat `score >= threshold` as dedup. | Return explicit `advisory=True` flag on `Recollection` from `near_duplicates()`; log when near-dup fired but fingerprint differed. |
| 23 | **pgvector extension requires superuser on some hosts.** `CREATE EXTENSION` may fail on managed RDS without prior enablement. | Document extension enablement in infra README; migration split: extension in Terraform, table in app migration. |

### 17.6 Priority roadmap (suggested)

| Phase | Work | Unblocks |
|-------|------|----------|
| **Now** | pgvector Postgres in CI; document `pgvector/pgvector` Docker image | Confidence in production DDL path |
| **Next** | Model-backed embedder + `embedder_version` column + rebuild runbook | Useful production recall |
| **Then** | Incremental rebuild + index/registry consistency check in orchestration | Operability at scale |
| **Later** | Multi-vector or DSL-resolved spec text; hydrated recall API | Richer agent context |

---

## Related documentation

- [Registry README](../registry/README.md) — immutable system of record
- [Registry DOCUMENTATION](../registry/DOCUMENTATION.md) — experiment/decision model in depth
- [CLAUDE.md](../../CLAUDE.md) — project-wide conventions and testing standard (§8)
