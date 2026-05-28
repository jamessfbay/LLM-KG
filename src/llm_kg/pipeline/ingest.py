from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.embeddings import EmbeddingClient, build_embedding_client
from llm_kg.llm import LLMClient
from llm_kg.models import Claim, EmbeddingRecord, Entity, Evidence, IngestResult, TextUnit, WikiPage
from llm_kg.models.core import utc_now
from llm_kg.pipeline.chunking import chunk_document
from llm_kg.readers import read_source
from llm_kg.readers.common import slugify, stable_id
from llm_kg.storage import JsonlStore, MarkdownStore, build_postgres_store


def ingest_source(
    path: Path,
    llm: LLMClient,
    workspace: Path,
    settings: Settings | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> IngestResult:
    workspace = workspace.resolve()
    settings = settings or Settings.from_env(workspace)
    document = read_source(path)
    text_units = chunk_document(document)
    wiki_page = llm.generate_wiki_page(document)
    claims, evidence, entities, relations = llm.extract_knowledge(document, wiki_page)

    markdown_store = MarkdownStore(workspace)
    markdown_store.save_page(wiki_page)
    for entity in entities:
        markdown_store.save_page(_entity_page(entity))
    markdown_store.update_index()
    markdown_store.append_log(
        f"ingested `{document.source_path}` as `{wiki_page.path}` with "
        f"{len(claims)} claims, {len(entities)} entities, {len(relations)} relations"
    )

    jsonl = JsonlStore(workspace)
    jsonl.upsert("claims.jsonl", claims)
    jsonl.upsert("evidence.jsonl", evidence)
    jsonl.upsert("nodes.jsonl", entities)
    jsonl.upsert("edges.jsonl", relations)

    postgres = build_postgres_store(settings)
    if postgres:
        embedder = embedding_client or build_embedding_client(settings)
        wiki_pages = [wiki_page, *(_entity_page(entity) for entity in entities)]
        postgres.upsert_ingest(
            document=document,
            text_units=text_units,
            wiki_pages=wiki_pages,
            claims=claims,
            evidence=evidence,
            entities=entities,
            relations=relations,
            embeddings=_build_embeddings(
                text_units=text_units,
                wiki_pages=wiki_pages,
                claims=claims,
                evidence=evidence,
                entities=entities,
                embedder=embedder,
            ),
        )

    return IngestResult(
        document=document,
        text_units=text_units,
        wiki_page=wiki_page,
        claims=claims,
        evidence=evidence,
        entities=entities,
        relations=relations,
    )


def _entity_page(entity: Entity) -> WikiPage:
    content = "\n".join(
        [
            f"# {entity.name}",
            "",
            "## Type",
            entity.entity_type,
            "",
            "## Description",
            entity.description or f"{entity.name} is an entity extracted from source material.",
            "",
            "## Source IDs",
            *(f"- `{source_id}`" for source_id in entity.source_ids),
            "",
        ]
    )
    return WikiPage(
        id=stable_id("wiki", f"entity:{entity.id}"),
        title=entity.name,
        page_type="entity",
        path=f"wiki/entities/{slugify(entity.name)}.md",
        content_md=content,
        source_ids=entity.source_ids,
        wikilinks=[],
        tags=[entity.entity_type],
        updated_at=utc_now(),
    )


def _build_embeddings(
    text_units: list[TextUnit],
    wiki_pages: list[WikiPage],
    claims: list[Claim],
    evidence: list[Evidence],
    entities: list[Entity],
    embedder: EmbeddingClient,
) -> list[EmbeddingRecord]:
    targets: list[tuple[str, str, str]] = []
    targets.extend(("text_unit", item.id, item.text) for item in text_units)
    targets.extend(("wiki_page", item.id, item.content_md) for item in wiki_pages)
    targets.extend(("claim", item.id, item.text) for item in claims)
    targets.extend(("evidence", item.id, item.quote) for item in evidence)
    targets.extend(
        ("entity", item.id, f"{item.name}\n{item.entity_type}\n{item.description or ''}") for item in entities
    )
    vectors = embedder.embed_texts([text for _, _, text in targets])
    return [
        EmbeddingRecord(
            id=stable_id("emb", f"{record_type}:{record_id}:{embedder.model}"),
            record_type=record_type,  # type: ignore[arg-type]
            record_id=record_id,
            embedding=vector,
            model=embedder.model,
            dimensions=embedder.dimensions,
            text=text,
        )
        for (record_type, record_id, text), vector in zip(targets, vectors)
    ]
