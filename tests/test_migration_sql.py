from pathlib import Path


def test_pgvector_migration_defines_core_tables() -> None:
    sql = Path("migrations/001_init_pgvector.sql").read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS documents" in sql
    assert "CREATE TABLE IF NOT EXISTS text_units" in sql
    assert "CREATE TABLE IF NOT EXISTS embeddings" in sql
    assert "VECTOR(1536)" in sql
