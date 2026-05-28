CREATE TABLE IF NOT EXISTS ontology_schemas (
    id TEXT PRIMARY KEY,
    entity_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    relation_predicates JSONB NOT NULL DEFAULT '[]'::jsonb,
    require_claim_evidence BOOLEAN NOT NULL DEFAULT true,
    require_relation_trace BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reasoning_traces (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    mode TEXT NOT NULL,
    hits JSONB NOT NULL DEFAULT '[]'::jsonb,
    used_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    used_relation_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    used_evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    reasoning_steps JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence DOUBLE PRECISION NOT NULL,
    decision_output TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reasoning_traces_created_at ON reasoning_traces(created_at);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_claim_ids ON reasoning_traces USING gin(used_claim_ids);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_relation_ids ON reasoning_traces USING gin(used_relation_ids);
CREATE INDEX IF NOT EXISTS idx_reasoning_traces_evidence_ids ON reasoning_traces USING gin(used_evidence_ids);
