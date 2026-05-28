from pathlib import Path


def test_pgvector_migration_defines_core_tables() -> None:
    sql = "\n".join(path.read_text(encoding="utf-8") for path in sorted(Path("migrations").glob("*.sql")))

    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS documents" in sql
    assert "CREATE TABLE IF NOT EXISTS text_units" in sql
    assert "CREATE TABLE IF NOT EXISTS embeddings" in sql
    assert "CREATE TABLE IF NOT EXISTS audit_events" in sql
    assert "CREATE TABLE IF NOT EXISTS update_proposals" in sql
    assert "ADD COLUMN IF NOT EXISTS review_state" in sql
    assert "VECTOR(1536)" in sql
