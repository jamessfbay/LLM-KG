from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_kg.config import Settings
from llm_kg.models import (
    AuditEvent,
    Claim,
    Document,
    EmbeddingRecord,
    Entity,
    Evidence,
    QueryHit,
    Relation,
    TextUnit,
    UpdateProposalDraft,
    WikiPage,
)
from llm_kg.readers.common import stable_id


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
                cur.execute(
                    """
                    SELECT to_regclass('public.documents'),
                           to_regclass('public.embeddings'),
                           to_regclass('public.audit_events'),
                           to_regclass('public.update_proposals')
                    """
                )
                documents_table, embeddings_table, audit_table, proposals_table = cur.fetchone()
                if not documents_table or not embeddings_table:
                    return {"database": database, "user": user, "migrated": False, "documents": 0, "embeddings": 0}
                cur.execute("SELECT count(*) FROM documents")
                documents = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM embeddings")
                embeddings = cur.fetchone()[0]
                audit_events = 0
                proposals = 0
                if audit_table:
                    cur.execute("SELECT count(*) FROM audit_events")
                    audit_events = cur.fetchone()[0]
                if proposals_table:
                    cur.execute("SELECT count(*) FROM update_proposals")
                    proposals = cur.fetchone()[0]
        return {
            "database": database,
            "user": user,
            "migrated": True,
            "documents": documents,
            "embeddings": embeddings,
            "audit_events": audit_events,
            "proposals": proposals,
        }

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
                          (id, title, page_type, path, content_md, source_ids, wikilinks, tags, updated_at,
                           review_state, version, created_by, updated_by, supersedes_id, superseded_by_id,
                           governance_notes)
                        VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          title=EXCLUDED.title,
                          page_type=EXCLUDED.page_type,
                          path=EXCLUDED.path,
                          content_md=EXCLUDED.content_md,
                          source_ids=EXCLUDED.source_ids,
                          wikilinks=EXCLUDED.wikilinks,
                          tags=EXCLUDED.tags,
                          updated_at=EXCLUDED.updated_at,
                          review_state=EXCLUDED.review_state,
                          version=EXCLUDED.version,
                          updated_by=EXCLUDED.updated_by,
                          supersedes_id=EXCLUDED.supersedes_id,
                          superseded_by_id=EXCLUDED.superseded_by_id,
                          governance_notes=EXCLUDED.governance_notes
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
                            page.review_state,
                            page.version,
                            page.created_by,
                            page.updated_by,
                            page.supersedes_id,
                            page.superseded_by_id,
                            page.governance_notes,
                        ),
                    )
                    _insert_audit_event(cur, "create", "wiki_page", page.id, None, _payload(page))
                for item in evidence:
                    cur.execute(
                        """
                        INSERT INTO evidence
                          (id, source_id, quote, page_number, url, section, confidence,
                           review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          quote=EXCLUDED.quote,
                          page_number=EXCLUDED.page_number,
                          url=EXCLUDED.url,
                          section=EXCLUDED.section,
                          confidence=EXCLUDED.confidence,
                          review_state=EXCLUDED.review_state,
                          version=EXCLUDED.version,
                          updated_by=EXCLUDED.updated_by,
                          updated_at=EXCLUDED.updated_at,
                          supersedes_id=EXCLUDED.supersedes_id,
                          superseded_by_id=EXCLUDED.superseded_by_id,
                          governance_notes=EXCLUDED.governance_notes
                        """,
                        (
                            item.id,
                            item.source_id,
                            item.quote,
                            item.page_number,
                            item.url,
                            item.section,
                            item.confidence,
                            item.review_state,
                            item.version,
                            item.created_by,
                            item.updated_by,
                            item.updated_at,
                            item.supersedes_id,
                            item.superseded_by_id,
                            item.governance_notes,
                        ),
                    )
                    _insert_audit_event(cur, "create", "evidence", item.id, None, _payload(item))
                for claim in claims:
                    cur.execute(
                        """
                        INSERT INTO claims
                          (id, text, source_ids, evidence_ids, subject, predicate, object, confidence, status,
                           created_at, review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes)
                        VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          text=EXCLUDED.text,
                          source_ids=EXCLUDED.source_ids,
                          evidence_ids=EXCLUDED.evidence_ids,
                          subject=EXCLUDED.subject,
                          predicate=EXCLUDED.predicate,
                          object=EXCLUDED.object,
                          confidence=EXCLUDED.confidence,
                          status=EXCLUDED.status,
                          review_state=EXCLUDED.review_state,
                          version=EXCLUDED.version,
                          updated_by=EXCLUDED.updated_by,
                          updated_at=EXCLUDED.updated_at,
                          supersedes_id=EXCLUDED.supersedes_id,
                          superseded_by_id=EXCLUDED.superseded_by_id,
                          governance_notes=EXCLUDED.governance_notes
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
                            claim.review_state,
                            claim.version,
                            claim.created_by,
                            claim.updated_by,
                            claim.updated_at,
                            claim.supersedes_id,
                            claim.superseded_by_id,
                            claim.governance_notes,
                        ),
                    )
                    _insert_audit_event(cur, "create", "claim", claim.id, None, _payload(claim))
                for entity in entities:
                    cur.execute(
                        """
                        INSERT INTO entities
                          (id, name, entity_type, aliases, description, source_ids,
                           review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes)
                        VALUES (%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          name=EXCLUDED.name,
                          entity_type=EXCLUDED.entity_type,
                          aliases=EXCLUDED.aliases,
                          description=EXCLUDED.description,
                          source_ids=EXCLUDED.source_ids,
                          review_state=EXCLUDED.review_state,
                          version=EXCLUDED.version,
                          updated_by=EXCLUDED.updated_by,
                          updated_at=EXCLUDED.updated_at,
                          supersedes_id=EXCLUDED.supersedes_id,
                          superseded_by_id=EXCLUDED.superseded_by_id,
                          governance_notes=EXCLUDED.governance_notes
                        """,
                        (
                            entity.id,
                            entity.name,
                            entity.entity_type,
                            _json(entity.aliases),
                            entity.description,
                            _json(entity.source_ids),
                            entity.review_state,
                            entity.version,
                            entity.created_by,
                            entity.updated_by,
                            entity.updated_at,
                            entity.supersedes_id,
                            entity.superseded_by_id,
                            entity.governance_notes,
                        ),
                    )
                    _insert_audit_event(cur, "create", "entity", entity.id, None, _payload(entity))
                for relation in relations:
                    cur.execute(
                        """
                        INSERT INTO relationships
                          (id, subject_id, predicate, object_id, claim_ids, evidence_ids, confidence, valid_from,
                           valid_to, review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes)
                        VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET
                          predicate=EXCLUDED.predicate,
                          claim_ids=EXCLUDED.claim_ids,
                          evidence_ids=EXCLUDED.evidence_ids,
                          confidence=EXCLUDED.confidence,
                          valid_from=EXCLUDED.valid_from,
                          valid_to=EXCLUDED.valid_to,
                          review_state=EXCLUDED.review_state,
                          version=EXCLUDED.version,
                          updated_by=EXCLUDED.updated_by,
                          updated_at=EXCLUDED.updated_at,
                          supersedes_id=EXCLUDED.supersedes_id,
                          superseded_by_id=EXCLUDED.superseded_by_id,
                          governance_notes=EXCLUDED.governance_notes
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
                            relation.review_state,
                            relation.version,
                            relation.created_by,
                            relation.updated_by,
                            relation.updated_at,
                            relation.supersedes_id,
                            relation.superseded_by_id,
                            relation.governance_notes,
                        ),
                    )
                    _insert_audit_event(cur, "create", "relation", relation.id, None, _payload(relation))
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

    def get_claim(self, claim_id: str) -> Claim | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, text, source_ids, evidence_ids, subject, predicate, object, confidence, status,
                           created_at, review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes
                    FROM claims WHERE id = %s
                    """,
                    (claim_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return Claim(
            id=row[0],
            text=row[1],
            source_ids=list(row[2] or []),
            evidence_ids=list(row[3] or []),
            subject=row[4],
            predicate=row[5],
            object=row[6],
            confidence=float(row[7]),
            status=row[8],
            created_at=row[9],
            review_state=row[10],
            version=row[11],
            created_by=row[12],
            updated_by=row[13],
            updated_at=row[14],
            supersedes_id=row[15],
            superseded_by_id=row[16],
            governance_notes=row[17],
        )

    def update_claim(self, claim: Claim, before: dict[str, Any] | None = None) -> str:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE claims SET
                      text=%s,
                      source_ids=%s::jsonb,
                      evidence_ids=%s::jsonb,
                      subject=%s,
                      predicate=%s,
                      object=%s,
                      confidence=%s,
                      status=%s,
                      review_state=%s,
                      version=%s,
                      updated_by=%s,
                      updated_at=%s,
                      supersedes_id=%s,
                      superseded_by_id=%s,
                      governance_notes=%s
                    WHERE id=%s
                    """,
                    (
                        claim.text,
                        _json(claim.source_ids),
                        _json(claim.evidence_ids),
                        claim.subject,
                        claim.predicate,
                        claim.object,
                        claim.confidence,
                        claim.status,
                        claim.review_state,
                        claim.version,
                        claim.updated_by,
                        claim.updated_at,
                        claim.supersedes_id,
                        claim.superseded_by_id,
                        claim.governance_notes,
                        claim.id,
                    ),
                )
                event_id = _insert_audit_event(cur, "apply", "claim", claim.id, before, _payload(claim))
            conn.commit()
        return event_id

    def get_evidence_many(self, evidence_ids: list[str]) -> list[Evidence]:
        if not evidence_ids:
            return []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, source_id, quote, page_number, url, section, confidence, review_state, version,
                           created_by, updated_by, updated_at, supersedes_id, superseded_by_id, governance_notes
                    FROM evidence WHERE id = ANY(%s)
                    """,
                    (evidence_ids,),
                )
                rows = cur.fetchall()
        return [
            Evidence(
                id=row[0],
                source_id=row[1],
                quote=row[2],
                page_number=row[3],
                url=row[4],
                section=row[5],
                confidence=float(row[6]),
                review_state=row[7],
                version=row[8],
                created_by=row[9],
                updated_by=row[10],
                updated_at=row[11],
                supersedes_id=row[12],
                superseded_by_id=row[13],
                governance_notes=row[14],
            )
            for row in rows
        ]

    def get_relations_for_claim(self, claim_id: str) -> list[Relation]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, subject_id, predicate, object_id, claim_ids, evidence_ids, confidence, valid_from,
                           valid_to, review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes
                    FROM relationships
                    WHERE claim_ids ? %s
                    """,
                    (claim_id,),
                )
                rows = cur.fetchall()
        return [_relation_from_row(row) for row in rows]

    def get_relation(self, relation_id: str) -> Relation | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, subject_id, predicate, object_id, claim_ids, evidence_ids, confidence, valid_from,
                           valid_to, review_state, version, created_by, updated_by, updated_at, supersedes_id,
                           superseded_by_id, governance_notes
                    FROM relationships WHERE id = %s
                    """,
                    (relation_id,),
                )
                row = cur.fetchone()
        return _relation_from_row(row) if row else None

    def get_entity(self, entity_id: str) -> Entity | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, entity_type, aliases, description, source_ids, review_state, version,
                           created_by, updated_by, updated_at, supersedes_id, superseded_by_id, governance_notes
                    FROM entities WHERE id = %s
                    """,
                    (entity_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return Entity(
            id=row[0],
            name=row[1],
            entity_type=row[2],
            aliases=list(row[3] or []),
            description=row[4],
            source_ids=list(row[5] or []),
            review_state=row[6],
            version=row[7],
            created_by=row[8],
            updated_by=row[9],
            updated_at=row[10],
            supersedes_id=row[11],
            superseded_by_id=row[12],
            governance_notes=row[13],
        )

    def upsert_proposal(self, proposal: UpdateProposalDraft) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO update_proposals
                      (id, proposal_type, target_type, target_id, title, rationale, evidence_ids,
                       source_signal_ids, proposed_change, confidence, status, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                      title=EXCLUDED.title,
                      rationale=EXCLUDED.rationale,
                      evidence_ids=EXCLUDED.evidence_ids,
                      source_signal_ids=EXCLUDED.source_signal_ids,
                      proposed_change=EXCLUDED.proposed_change,
                      confidence=EXCLUDED.confidence,
                      status=EXCLUDED.status
                    """,
                    (
                        proposal.id,
                        proposal.proposal_type,
                        proposal.target_type,
                        proposal.target_id,
                        proposal.title,
                        proposal.rationale,
                        _json(proposal.evidence_ids),
                        _json(proposal.source_signal_ids),
                        _json(proposal.proposed_change),
                        proposal.confidence,
                        proposal.status,
                        proposal.created_at,
                    ),
                )
                _insert_audit_event(cur, "proposal", proposal.target_type, proposal.target_id or proposal.id, None, _payload(proposal))
            conn.commit()

    def get_proposal(self, proposal_id: str) -> UpdateProposalDraft | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, proposal_type, target_type, target_id, title, rationale, evidence_ids,
                           source_signal_ids, proposed_change, confidence, status, created_at
                    FROM update_proposals WHERE id = %s
                    """,
                    (proposal_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return UpdateProposalDraft(
            id=row[0],
            proposal_type=row[1],
            target_type=row[2],
            target_id=row[3],
            title=row[4],
            rationale=row[5],
            evidence_ids=list(row[6] or []),
            source_signal_ids=list(row[7] or []),
            proposed_change=dict(row[8] or {}),
            confidence=float(row[9]),
            status=row[10],
            created_at=row[11],
        )

    def insert_audit_event(self, event: AuditEvent) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                _insert_audit_event(
                    cur,
                    event.event_type,
                    event.target_type,
                    event.target_id,
                    event.before,
                    event.after,
                    event.actor,
                    event.source,
                    event.id,
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


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _insert_audit_event(
    cur,
    event_type: str,
    target_type: str,
    target_id: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    actor: str = "system",
    source: str = "llm-kg",
    event_id: str | None = None,
) -> str:
    event_id = event_id or stable_id(
        "audit",
        json.dumps(
            {
                "event_type": event_type,
                "target_type": target_type,
                "target_id": target_id,
                "after": after,
                "actor": actor,
                "source": source,
            },
            sort_keys=True,
            default=str,
        ),
    )
    cur.execute(
        """
        INSERT INTO audit_events
          (id, event_type, target_type, target_id, actor, source, before_payload, after_payload)
        VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
        ON CONFLICT (id) DO NOTHING
        """,
        (event_id, event_type, target_type, target_id, actor, source, _json(before), _json(after)),
    )
    return event_id


def _relation_from_row(row) -> Relation:
    return Relation(
        id=row[0],
        subject_id=row[1],
        predicate=row[2],
        object_id=row[3],
        claim_ids=list(row[4] or []),
        evidence_ids=list(row[5] or []),
        confidence=float(row[6]),
        valid_from=row[7],
        valid_to=row[8],
        review_state=row[9],
        version=row[10],
        created_by=row[11],
        updated_by=row[12],
        updated_at=row[13],
        supersedes_id=row[14],
        superseded_by_id=row[15],
        governance_notes=row[16],
    )


def _vector(value: list[float]) -> str:
    return "[" + ",".join(f"{item:.8f}" for item in value) + "]"


def build_postgres_store(settings: Settings) -> PostgresStore | None:
    if not settings.database_url:
        return None
    return PostgresStore(settings.database_url)
