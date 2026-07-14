-- Registry: append-only, event-sourced, tamper-evident system of record.
--
-- One unified event log is the source of truth. Each row is a record of one of
-- six kinds; the per-stream hash chain (prev_hash -> record_hash) spans the
-- Experiment -> Result -> Decision events of an experiment stream and the
-- Strategy -> StrategyVersion -> LifecycleEvent events of a strategy stream.
-- A single log keeps the chain coherent and makes verify_chain a linear scan.

CREATE TABLE IF NOT EXISTS registry_events (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    stream_id       TEXT        NOT NULL,
    stream_kind     TEXT        NOT NULL
        CHECK (stream_kind IN ('experiment', 'strategy')),
    seq             BIGINT      NOT NULL CHECK (seq >= 0),
    record_kind     TEXT        NOT NULL
        CHECK (record_kind IN (
            'experiment', 'result', 'decision',
            'strategy', 'strategy_version', 'lifecycle_event'
        )),
    record_id       TEXT        NOT NULL,
    payload         JSONB       NOT NULL,
    prev_hash       TEXT,
    record_hash     TEXT        NOT NULL,
    knowledge_time  TIMESTAMPTZ NOT NULL,
    -- Per-stream chain integrity: positions are dense and unique per stream, so
    -- a concurrent racing append fails the unique check and retries (no lost write).
    CONSTRAINT registry_events_stream_seq_unique UNIQUE (stream_id, seq),
    CONSTRAINT registry_events_record_id_unique UNIQUE (record_id),
    -- seq 0 opens a stream (no prior link); every later record must be chained.
    CONSTRAINT registry_events_genesis_unchained
        CHECK ((seq = 0) = (prev_hash IS NULL))
);

CREATE INDEX IF NOT EXISTS registry_events_stream_idx
    ON registry_events (stream_id, seq);
CREATE INDEX IF NOT EXISTS registry_events_kind_idx
    ON registry_events (record_kind);

-- Query-path expression indexes (author / niche / outcome / lineage / dedup).
CREATE INDEX IF NOT EXISTS registry_events_author_idx
    ON registry_events ((payload ->> 'author'));
CREATE INDEX IF NOT EXISTS registry_events_niche_idx
    ON registry_events ((payload ->> 'niche'));
CREATE INDEX IF NOT EXISTS registry_events_fingerprint_idx
    ON registry_events ((payload ->> 'trial_fingerprint'));
CREATE INDEX IF NOT EXISTS registry_events_parent_idx
    ON registry_events ((payload ->> 'parent_experiment_id'));
CREATE INDEX IF NOT EXISTS registry_events_outcome_idx
    ON registry_events ((payload ->> 'outcome'));

-- Append-only enforcement: reject UPDATE and DELETE at the database layer so a
-- record can never be quietly rewritten, even by a privileged client.
CREATE OR REPLACE FUNCTION registry_reject_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'registry is append-only: UPDATE and DELETE are forbidden on %', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS registry_events_no_update ON registry_events;
CREATE OR REPLACE TRIGGER registry_events_no_update
    BEFORE UPDATE ON registry_events
    FOR EACH ROW EXECUTE FUNCTION registry_reject_mutation();

DROP TRIGGER IF EXISTS registry_events_no_delete ON registry_events;
CREATE OR REPLACE TRIGGER registry_events_no_delete
    BEFORE DELETE ON registry_events
    FOR EACH ROW EXECUTE FUNCTION registry_reject_mutation();
