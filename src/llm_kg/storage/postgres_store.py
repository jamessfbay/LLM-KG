from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_kg.config import Settings
from llm_kg.models import Claim, Document, EmbeddingRecord, Entity, Evidence, QueryHit, Relation, TextUnit, WikiPage


class PostgresStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Postgres support requires psycopg[binary].") from exc
        return psycopg.connect(self.database_url)

    def status(self) -> dict[str, Any]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), current_user")
                database, user = cur.fetchone()
                cur.execute("SELECT to_regclass('public.documents'), to_regclass('public.embeddings')")
                documents_table, embeddings_table = cur.fetchone()
                if not documents_table or not embeddings_table:
                    return {"database": database, "user": user, "migrated": False, "documents": 0, "embeddings": 0}
                cur.execute("SELECT count(*) FROM documents")
                documents = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM embeddings")
                embeddings = cur.fetchone()[0]
        return {"database": database, "user": user, "migrated": True, "documents": documents, "embeddings": embeddings}

    def apply_migrations(self, migrations_dir: Path) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                for path in sorted(migrations_dir.glob("*.sql")):
                    cur.execute(path.read_text(encoding="utf-8"))
            conn.commit()

    def upsert_ingest(
        self,
        document: Document,
        text_units: list[TextUnit],
        wiki_pages: list[WikiPage],
        claims: list[Claim],
        evidence: list[Evidence],
        entities: list[Entity],
        relations: list[Relation],
        embeddings: list[EmbeddingRecord],
    ) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                      (id, title, source_path, source_type, content, author, created_at, ingested_at, hash, metadata)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                      title=EXCLUDED.title,
                      source_path=EXCLUDED.source_path,
                      source_type=EXCLUDED.source_type,
                      content=EXCLUDED.content,
                      hash=EXCLUDED.hash
                    """,
                    (
                        document.id,
                        document.title,
                        document.source_path,
                        document.source_type,
                        document.content,
                        document.author,
                        document.created_at,
                        document.ingested_at,
                        document.hash,
                        "{}",
                    ),
                )
                for text_unit in text_units:
                    cur.execute(
                        """
                        INSERT INTO text_units
                          (id, document_id, chunk_index, text, start_char, end_char, token_count, source_ids)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                          text=EXCLUDED.text,
                          token_count=EXCLUDED.token_count,
                          source_ids=EXCLUDED.source_ids
                        """,
                        (
                            text_unit.id,
                            text_unit.document_id,
                            text_unit.chunk_index,
                            text_unit.text,
                            text_unit.start_char,
                            text_unit.end_char,
                            text_unit.token_count,
                            _json(text_unit.source_ids),
                        ),
                    )
                for page in wiki_pages:
                    cur.execute(
                        """
                        INSERT INTO wiki_pages
                          (id, title, page_type, path, content_md, source_ids, wikilinks, tags, updated_at)
                        VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          title=EXCLUDED.title,
                          page_type=EXCLUDED.page_type,
                          path=EXCLUDED.path,
                          content_md=EXCLUDED.content_md,
                          source_ids=EXCLUDED.source_ids,
                          wikilinks=EXCLUDED.wikilinks,
                          tags=EXCLUDED.tags,
                          updated_at=EXCLUDED.updated_at
                        """,
                        (
                            page.id,
                            page.title,
                            page.page_type,
                            page.path,
                            page.content_md,
                            _json(page.source_ids),
                            _json(page.wikilinks),
                            _json(page.tags),
                            page.updated_at,
                        ),
                    )
                for item in evidence:
                    cur.execute(
                        """
                        INSERT INTO evidence
                          (id, source_id, quote, page_number, url, section, confidence)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          quote=EXCLUDED.quote,
                          page_number=EXCLUDED.page_number,
                          url=EXCLUDED.url,
                          section=EXCLUDED.section,
                          confidence=EXCLUDED.confidence
                        """,
                        (item.id, item.source_id, item.quote, item.page_number, item.url, item.section, item.confidence),
                    )
                for claim in claims:
                    cur.execute(
                        """
                        INSERT INTO claims
                          (id, text, source_ids, evidence_ids, subject, predicate, object, confidence, status, created_at)
                        VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          text=EXCLUDED.text,
                          source_ids=EXCLUDED.source_ids,
                          evidence_ids=EXCLUDED.evidence_ids,
                          subject=EXCLUDED.subject,
                          predicate=EXCLUDED.predicate,
                          object=EXCLUDED.object,
                          confidence=EXCLUDED.confidence,
                          status=EXCLUDED.status
                        """,
                        (
                            claim.id,
                            claim.text,
                            _json(claim.source_ids),
                            _json(claim.evidence_ids),
                            claim.subject,
                            claim.predicate,
                            claim.object,
                            claim.confidence,
                            claim.status,
                            claim.created_at,
                        ),
                    )
                for entity in entities:
                    cur.execute(
                        """
                        INSERT INTO entities
                          (id, name, entity_type, aliases, description, source_ids)
                        VALUES (%s,%s,%s,%s::jsonb,%s,%s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                          name=EXCLUDED.name,
                          entity_type=EXCLUDED.entity_type,
                          aliases=EXCLUDED.aliases,
                          description=EXCLUDED.description,
                          source_ids=EXCLUDED.source_ids
                        """,
                        (
                            entity.id,
                            entity.name,
                            entity.entity_type,
                            _json(entity.aliases),
                            entity.description,
                            _json(entity.source_ids),
                        ),
                    )
                for relation in relations:
                    cur.execute(
                        """
                        INSERT INTO relationships
                          (id, subject_id, predicate, object_id, claim_ids, evidence_ids, confidence, valid_from, valid_to)
                        VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          predicate=EXCLUDED.predicate,
                          claim_ids=EXCLUDED.claim_ids,
                          evidence_ids=EXCLUDED.evidence_ids,
                          confidence=EXCLUDED.confidence,
                          valid_from=EXCLUDED.valid_from,
                          valid_to=EXCLUDED.valid_to
                        """,
                        (
                            relation.id,
                            relation.subject_id,
                            relation.predicate,
                            relation.object_id,
                            _json(relation.claim_ids),
                            _json(relation.evidence_ids),
                            relation.confidence,
                            relation.valid_from,
                            relation.valid_to,
                        ),
                    )
                for embedding in embeddings:
                    cur.execute(
                        """
                        INSERT INTO embeddings
                          (id, record_type, record_id, embedding, model, dimensions, text, created_at)
                        VALUES (%s,%s,%s,%s::vector,%s,%s,%s,%s)
                        ON CONFLICT (record_type, record_id) DO UPDATE SET
                          embedding=EXCLUDED.embedding,
                          model=EXCLUDED.model,
                          dimensions=EXCLUDED.dimensions,
                          text=EXCLUDED.text,
                          created_at=EXCLUDED.created_at
                        """,
                        (
                            embedding.id,
                            embedding.record_type,
                            embedding.record_id,
                            _vector(embedding.embedding),
                            embedding.model,
                            embedding.dimensions,
                            embedding.text,
                            embedding.created_at,
                        ),
                    )
            conn.commit()

    def search_basic(self, query: str, query_embedding: list[float], top_k: int) -> list[QueryHit]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tu.id, tu.text,
                           (1 - (em.embedding <=> %s::vector)) AS vector_score,
                           ts_rank(tu.search_vector, plainto_tsquery('english', %s)) AS text_score
                    FROM text_units tu
                    JOIN embeddings em ON em.record_type = 'text_unit' AND em.record_id = tu.id
                    WHERE tu.search_vector @@ plainto_tsquery('english', %s)
                       OR em.embedding <=> %s::vector < 1.0
                    ORDER BY ((1 - (em.embedding <=> %s::vector)) + ts_rank(tu.search_vector, plainto_tsquery('english', %s))) DESC
                    LIMIT %s
                    """,
                    (
                        _vector(query_embedding),
                        query,
                        query,
                        _vector(query_embedding),
                        _vector(query_embedding),
                        query,
                        top_k,
                    ),
                )
                rows = cur.fetchall()
        return [
            QueryHit(kind="text_unit", id=row[0], text=row[1], score=round(float(row[2] or 0) + float(row[3] or 0), 4))
            for row in rows
        ]

    def search_local(self, query: str, query_embedding: list[float], top_k: int) -> tuple[list[QueryHit], list[Evidence]]:
        hits: list[QueryHit] = []
        evidence_items: list[Evidence] = []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.id, e.name, e.entity_type, coalesce(e.description, ''),
                           (1 - (em.embedding <=> %s::vector)) AS vector_score,
                           ts_rank(e.search_vector, plainto_tsquery('english', %s)) AS text_score
                    FROM entities e
                    LEFT JOIN embeddings em ON em.record_type = 'entity' AND em.record_id = e.id
                    WHERE e.search_vector @@ plainto_tsquery('english', %s)
                       OR e.name ILIKE ('%%' || %s || '%%')
                       OR (em.embedding IS NOT NULL AND em.embedding <=> %s::vector < 1.0)
                    ORDER BY (coalesce(1 - (em.embedding <=> %s::vector), 0) + ts_rank(e.search_vector, plainto_tsquery('english', %s))) DESC
                    LIMIT %s
                    """,
                    (_vector(query_embedding), query, query, query, _vector(query_embedding), _vector(query_embedding), query, top_k),
                )
                entity_rows = cur.fetchall()
                entity_ids = [row[0] for row in entity_rows]
                for row in entity_rows:
                    hits.append(
                        QueryHit(
                            kind="entity",
                            id=row[0],
                            title=row[1],
                            text=f"{row[1]} ({row[2]}): {row[3]}",
                            score=round(float(row[4] or 0) + float(row[5] or 0), 4),
                        )
                    )
                if entity_ids:
                    cur.execute(
                        """
                        SELECT r.id, s.name, r.predicate, o.name, r.claim_ids, r.evidence_ids, r.confidence
                        FROM relationships r
                        JOIN entities s ON s.id = r.subject_id
                        JOIN entities o ON o.id = r.object_id
                        WHERE r.subject_id = ANY(%s) OR r.object_id = ANY(%s)
                        ORDER BY r.confidence DESC
                        LIMIT %s
                        """,
                        (entity_ids, entity_ids, top_k * 2),
                    )
                    relation_rows = cur.fetchall()
                    evidence_ids: set[str] = set()
                    claim_ids: set[str] = set()
                    for row in relation_rows:
                        row_claim_ids = list(row[4] or [])
                        row_evidence_ids = list(row[5] or [])
                        claim_ids.update(row_claim_ids)
                        evidence_ids.update(row_evidence_ids)
                        hits.append(
                            QueryHit(
                                kind="relation",
                                id=row[0],
                                title=row[2],
                                text=f"{row[1]} -> {row[2]} -> {row[3]}",
                                score=float(row[6] or 0),
                                evidence_ids=row_evidence_ids,
                            )
                        )
                    if claim_ids:
                        cur.execute(
                            """
                            SELECT id, text, evidence_ids, confidence
                            FROM claims
                            WHERE id = ANY(%s)
                            ORDER BY confidence DESC
                            LIMIT %s
                            """,
                            (list(claim_ids), top_k * 2),
                        )
                        for row in cur.fetchall():
                            row_evidence_ids = list(row[2] or [])
                            evidence_ids.update(row_evidence_ids)
                            hits.append(
                                QueryHit(
                                    kind="claim",
                                    id=row[0],
                                    text=row[1],
                                    score=float(row[3] or 0),
                                    evidence_ids=row_evidence_ids,
                                )
                            )
                    if evidence_ids:
                        cur.execute(
                            """
                            SELECT id, source_id, quote, page_number, url, section, confidence
                            FROM evidence
                            WHERE id = ANY(%s)
                            ORDER BY confidence DESC
                            LIMIT %s
                            """,
                            (list(evidence_ids), top_k * 3),
                        )
                        for row in cur.fetchall():
                            item = Evidence(
                                id=row[0],
                                source_id=row[1],
                                quote=row[2],
                                page_number=row[3],
                                url=row[4],
                                section=row[5],
                                confidence=float(row[6]),
                            )
                            evidence_items.append(item)
                            hits.append(
                                QueryHit(
                                    kind="evidence",
                                    id=item.id,
                                    text=item.quote,
                                    score=item.confidence,
                                    evidence_ids=[item.id],
                                )
                            )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k], evidence_items


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def _vector(value: list[float]) -> str:
    return "[" + ",".join(f"{item:.8f}" for item in value) + "]"


def build_postgres_store(settings: Settings) -> PostgresStore | None:
    if not settings.database_url:
        return None
    return PostgresStore(settings.database_url)
