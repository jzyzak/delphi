-- Agent memory derived index (rebuildable from registry_events).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_index (
    experiment_id       TEXT PRIMARY KEY,
    niche               TEXT        NOT NULL,
    outcome             TEXT        NOT NULL
        CHECK (outcome IN ('promoted', 'rejected', 'abandoned', 'pending')),
    trial_fingerprint   TEXT        NOT NULL,
    embedded_text       TEXT        NOT NULL,
    lessons             TEXT        NOT NULL DEFAULT '',
    embedding           vector({embedding_dim}) NOT NULL,
    knowledge_time      TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS memory_index_niche_idx
    ON memory_index (niche);

CREATE INDEX IF NOT EXISTS memory_index_outcome_idx
    ON memory_index (outcome);

CREATE INDEX IF NOT EXISTS memory_index_embedding_idx
    ON memory_index USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
