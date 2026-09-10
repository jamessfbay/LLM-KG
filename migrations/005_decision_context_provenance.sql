ALTER TABLE evidence
  ADD COLUMN IF NOT EXISTS source_content_hash TEXT,
  ADD COLUMN IF NOT EXISTS quote_start INTEGER,
  ADD COLUMN IF NOT EXISTS quote_end INTEGER,
  ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS extractor_version TEXT;

ALTER TABLE claims
  ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS conflicts_with JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE evidence
  DROP CONSTRAINT IF EXISTS evidence_quote_span_check;
ALTER TABLE evidence
  ADD CONSTRAINT evidence_quote_span_check CHECK (
    (quote_start IS NULL AND quote_end IS NULL)
    OR (quote_start >= 0 AND quote_end > quote_start)
  );
