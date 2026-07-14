-- PIT bitemporal store: initial schema
-- Append-only facts and universe membership with as-of query indexes.

CREATE TABLE IF NOT EXISTS pit_facts (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset         TEXT        NOT NULL,
    entity_id       TEXT        NOT NULL,
    effective_time  TIMESTAMPTZ NOT NULL,
    knowledge_time  TIMESTAMPTZ NOT NULL,
    values          JSONB       NOT NULL DEFAULT '{}',
    inserted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pit_facts_knowledge_gte_effective
        CHECK (knowledge_time >= effective_time),
    CONSTRAINT pit_facts_unique_version
        UNIQUE (dataset, entity_id, effective_time, knowledge_time)
);

-- Serves DISTINCT ON as-of: filter by dataset/entity/effective, pick latest knowledge <= T
CREATE INDEX IF NOT EXISTS pit_facts_as_of_idx
    ON pit_facts (dataset, entity_id, effective_time, knowledge_time DESC);

CREATE TABLE IF NOT EXISTS pit_universe (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    universe        TEXT        NOT NULL,
    entity_id       TEXT        NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('active', 'withdrawn')),
    effective_time  TIMESTAMPTZ NOT NULL,
    knowledge_time  TIMESTAMPTZ NOT NULL,
    values          JSONB       NOT NULL DEFAULT '{}',
    inserted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pit_universe_unique_version
        UNIQUE (universe, entity_id, effective_time, knowledge_time)
);

CREATE INDEX IF NOT EXISTS pit_universe_as_of_idx
    ON pit_universe (universe, entity_id, effective_time, knowledge_time DESC);

-- Append-only enforcement: reject UPDATE and DELETE at the database layer.
CREATE OR REPLACE FUNCTION pit_reject_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'PIT store is append-only: UPDATE and DELETE are forbidden on %', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER pit_facts_no_update
    BEFORE UPDATE ON pit_facts
    FOR EACH ROW EXECUTE FUNCTION pit_reject_mutation();

CREATE OR REPLACE TRIGGER pit_facts_no_delete
    BEFORE DELETE ON pit_facts
    FOR EACH ROW EXECUTE FUNCTION pit_reject_mutation();

CREATE OR REPLACE TRIGGER pit_universe_no_update
    BEFORE UPDATE ON pit_universe
    FOR EACH ROW EXECUTE FUNCTION pit_reject_mutation();

CREATE OR REPLACE TRIGGER pit_universe_no_delete
    BEFORE DELETE ON pit_universe
    FOR EACH ROW EXECUTE FUNCTION pit_reject_mutation();
