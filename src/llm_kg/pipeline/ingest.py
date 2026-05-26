from __future__ import annotations

from pathlib import Path

from llm_kg.llm import LLMClient
from llm_kg.models import Entity, IngestResult, WikiPage
from llm_kg.models.core import utc_now
from llm_kg.readers.common import slugify, stable_id
from llm_kg.readers import read_source
from llm_kg.storage import JsonlStore, MarkdownStore


def ingest_source(path: Path, llm: LLMClient, workspace: Path) -> IngestResult:
    workspace = workspace.resolve()
    document = read_source(path)
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

    return IngestResult(
        document=document,
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
