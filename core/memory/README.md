# Agent Memory

> **Full documentation:** [DOCUMENTATION.md](DOCUMENTATION.md) — every feature, model,
> index primitive, recall API, and design decision in depth. This page is the one-page
> quick start.

> **Scope:** derived semantic recall over the experiment registry. The registry
> ([registry/README.md](../registry/README.md)) remains the immutable source
> of truth; memory is a rebuildable index for agent context assembly (prompt 11).

Agent memory answers: *"Have we tried something like this?"* and *"What did we
learn in this niche?"* — including from **failures**. It indexes registry
experiments with embeddings and serves filtered recall; it does not generate
strategies or write experiments.

## Derived index, not authoritative

Memory is a **cache**. If the index is lost or stale, `rebuild_from_registry()`
reconstructs it from canonical registry records. Retrospective-evaluation
reproducibility never depends on the embedding index.

```python
from core.registry.store import InMemoryRegistryStore
from core.memory import (
    DeterministicEmbedder,
    InMemoryVectorIndex,
    MemoryRecall,
)

store = InMemoryRegistryStore()
embedder = DeterministicEmbedder()
index = InMemoryVectorIndex(store, embedder)
recall = MemoryRecall(embedder, index, store)

# After registry writes:
index.index(store.get_experiment(exp_id))
# Or full rebuild:
index.rebuild_from_registry()

hits = recall.recall(
    query="US election polling drift",
    niche="us_elections",
    outcome="any",  # failures included
    k=5,
)
```

## Semantic recall vs fingerprint accounting

| Mechanism | Role | Authoritative? |
|-----------|------|----------------|
| `trial_fingerprint` | Exact trial identity for honest counting | **Yes** (registry 03 / trials 06) |
| `near_duplicates()` | Semantic "looks close" warning for agents | **No — advisory only** |

Near-duplicate recall saves agent budget and improves hypotheses; it **never**
decides what counts as a trial.

## Recall contract

```python
recall.recall(query=..., niche=None, outcome="any", k=10) -> list[Recollection]
recall.lessons(query=..., niche=None, k=10) -> list[str]
recall.near_duplicates(spec_description=..., threshold=0.85) -> list[Recollection]
```

`Recollection` carries `experiment_id`, `niche`, `outcome` (`promoted` /
`rejected` / `abandoned` / `pending`), `score`, `embedded_text`, `lessons`, and
`trial_fingerprint`.

**Outcome mapping:** registry decisions use `promote` / `reject` / `abandon`;
the index exposes `promoted` / `rejected` / `abandoned` for recall filters.

**Lessons derivation:** the registry has no native lessons field. Memory derives
lessons from the latest `Decision.rationale` + `Decision.evidence` and key
`Result.metrics` at index time.

**Embedded text:** hypothesis, economic rationale, deterministic spec description
(`spec_kind`, `spec_hash`, canonical `params`), outcome, and lessons.

## Backends

| Backend | Use |
|---------|-----|
| `InMemoryVectorIndex` | Offline tests, local dev (numpy cosine) |
| `PostgresVectorIndex` | Production index (`pgvector`, table `memory_index`) |

Postgres connects via `DELPHI_PG_DSN` (same convention as registry/PIT).

## Embedders

| Embedder | Use |
|----------|-----|
| `DeterministicEmbedder` | Offline floor: hash n-gram vectors, reproducible, no network (default) |
| `BedrockEmbedder` | Production: Amazon Titan Text Embeddings V2 via `common.llm.embedding.BedrockEmbeddingClient` |

Select via `common.composition.build_embedder(settings)`: it returns the
deterministic floor unless `DELPHI_MODEL_EMBEDDING` is set, in which case it
builds `BedrockEmbedder` at `DELPHI_EMBEDDING_DIM`. The **embedding dimension is
the single source of truth** for the pgvector column — the `memory_index`
migration renders `vector({embedding_dim})` from `Embedder.dim`, so the column
always matches the stored vectors. Titan V2 accepts only 256/512/1024; the
deterministic floor uses 128. Dimension is fixed at first migration; changing it
later requires a fresh index rebuild.

## Testing

```bash
uv run pytest tests/memory
DELPHI_PG_DSN=postgresql://user:pass@localhost:5432/delphi \
  uv run pytest tests/memory -m postgres
```

Component tests M1–M7 use a deterministic fixture embedder; no network.

## Setup status

| Mode | Ready? | Notes |
|------|--------|-------|
| In-memory (tests, local agents) | **Yes** | `uv sync` only; no external services |
| Postgres + pgvector (production) | **Partial** | Requires pgvector-enabled Postgres + `DELPHI_PG_DSN`; CI does not run postgres tests yet |

See [DOCUMENTATION.md §16–§17](DOCUMENTATION.md#16-setup-and-operational-status) for the full setup checklist, open items, and improvement roadmap.

## Out of scope (this module)

- Orchestration scheduling / auto-rebuild jobs (Prompt 16)
- Writing experiments (registry's job)
- Trials accounting math (Prompt 06)
