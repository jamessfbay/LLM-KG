import os
from pathlib import Path

import pytest

from llm_kg.embeddings.mock_client import MockEmbeddingClient
from llm_kg.models import Claim, Document, Entity, Evidence, WikiPage
from llm_kg.pipeline.chunking import chunk_document
from llm_kg.pipeline.ingest import _build_embeddings
from llm_kg.storage.postgres_store import PostgresStore


@pytest.mark.skipif(not os.getenv("LLM_KG_TEST_DATABASE_URL"), reason="LLM_KG_TEST_DATABASE_URL not set")
def test_postgres_store_migrates_upserts_and_searches() -> None:
    store = PostgresStore(os.environ["LLM_KG_TEST_DATABASE_URL"])
    store.apply_migrations(Path("migrations"))

    doc = Document(
        id="doc_test_pg",
        title="Postgres Test Source",
        source_path="postgres-test.md",
        source_type="md",
        content="SB 330 affects Housing Project Alpha.",
        hash="hash_test_pg",
    )
    text_units = chunk_document(doc)
    page = WikiPage(
        id="wiki_test_pg",
        title="Postgres Test Source",
        page_type="source",
        path="wiki/sources/postgres-test.md",
        content_md="# Postgres Test Source\n\nSB 330 affects [[Housing Project Alpha]].",
        source_ids=[doc.id],
        wikilinks=["Housing Project Alpha"],
    )
    evidence = [Evidence(id="ev_test_pg", source_id=doc.id, quote=doc.content, confidence=0.9)]
    claims = [
        Claim(
            id="claim_test_pg",
            text=doc.content,
            source_ids=[doc.id],
            evidence_ids=["ev_test_pg"],
            subject="SB 330",
            predicate="affects",
            object="Housing Project Alpha",
            confidence=0.9,
        )
    ]
    entities = [
        Entity(id="entity_test_sb330", name="SB 330", entity_type="policy", source_ids=[doc.id]),
        Entity(id="entity_test_alpha", name="Housing Project Alpha", entity_type="project", source_ids=[doc.id]),
    ]
    relations = []
    embedder = MockEmbeddingClient()

    store.upsert_ingest(
        document=doc,
        text_units=text_units,
        wiki_pages=[page],
        claims=claims,
        evidence=evidence,
        entities=entities,
        relations=relations,
        embeddings=_build_embeddings(text_units, [page], claims, evidence, entities, embedder),
    )

    hits = store.search_basic("Housing Project Alpha", embedder.embed_text("Housing Project Alpha"), top_k=3)

    assert hits
    assert store.status()["migrated"] is True
