CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS ingest_runs (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    content TEXT NOT NULL,
    author TEXT,
    created_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS text_units (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    source_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_text_units_document_id ON text_units(document_id);
CREATE INDEX IF NOT EXISTS idx_text_units_search_vector ON text_units USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_text_units_text_trgm ON text_units USING gin(text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS wiki_pages (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    page_type TEXT NOT NULL,
    path TEXT NOT NULL,
    content_md TEXT NOT NULL,
    source_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    wikilinks JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    quote TEXT NOT NULL,
    page_number INTEGER,
    url TEXT,
    section TEXT,
    confidence DOUBLE PRECISION NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(quote, ''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_evidence_source_id ON evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_search_vector ON evidence USING gin(search_vector);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    confidence DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_claims_search_vector ON claims USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_claims_subject_trgm ON claims USING gin(subject gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_claims_object_trgm ON claims USING gin(object gin_trgm_ops);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    source_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (
      to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_entities_name_lower ON entities(lower(name));
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_search_vector ON entities USING gin(search_vector);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence DOUBLE PRECISION NOT NULL,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_relationships_subject_id ON relationships(subject_id);
CREATE INDEX IF NOT EXISTS idx_relationships_object_id ON relationships(object_id);
CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(predicate);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE(record_type, record_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_record ON embeddings(record_type, record_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
