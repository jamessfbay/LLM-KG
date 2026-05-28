ALTER TABLE wiki_pages
  ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'auto_accepted',
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS created_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS supersedes_id TEXT,
  ADD COLUMN IF NOT EXISTS superseded_by_id TEXT,
  ADD COLUMN IF NOT EXISTS governance_notes TEXT;

ALTER TABLE evidence
  ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'auto_accepted',
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS created_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS supersedes_id TEXT,
  ADD COLUMN IF NOT EXISTS superseded_by_id TEXT,
  ADD COLUMN IF NOT EXISTS governance_notes TEXT;

ALTER TABLE claims
  ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'auto_accepted',
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS created_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS supersedes_id TEXT,
  ADD COLUMN IF NOT EXISTS superseded_by_id TEXT,
  ADD COLUMN IF NOT EXISTS governance_notes TEXT;

ALTER TABLE entities
  ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'auto_accepted',
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS created_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS supersedes_id TEXT,
  ADD COLUMN IF NOT EXISTS superseded_by_id TEXT,
  ADD COLUMN IF NOT EXISTS governance_notes TEXT;

ALTER TABLE relationships
  ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'auto_accepted',
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS created_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_by TEXT NOT NULL DEFAULT 'system',
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS supersedes_id TEXT,
  ADD COLUMN IF NOT EXISTS superseded_by_id TEXT,
  ADD COLUMN IF NOT EXISTS governance_notes TEXT;

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    source TEXT NOT NULL,
    before_payload JSONB,
    after_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_events_target ON audit_events(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events(created_at);

CREATE TABLE IF NOT EXISTS update_proposals (
    id TEXT PRIMARY KEY,
    proposal_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_signal_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    proposed_change JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
