-- Runs once on first DB init (docker-entrypoint-initdb.d). The core/memory layer
-- stores embeddings in a pgvector column and calls register_vector on connect,
-- which requires the `vector` type to exist. The pgvector/pgvector image ships
-- the extension; this enables it in the `delphi` database.
CREATE EXTENSION IF NOT EXISTS vector;
