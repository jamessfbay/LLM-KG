from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.embeddings import EmbeddingClient, build_embedding_client
from llm_kg.llm import LLMClient
from llm_kg.governance import create_proposal
from llm_kg.models import Claim, Document, EmbeddingRecord, Entity, Evidence, IngestResult, Relation, TextUnit, WikiPage
from llm_kg.models.core import utc_now
from llm_kg.ontology import build_ontology_registry
from llm_kg.pipeline.chunking import chunk_document
from llm_kg.readers import read_source
from llm_kg.readers.common import slugify, stable_id
from llm_kg.storage import JsonlStore, MarkdownStore, build_postgres_store
from llm_kg.wiki import build_compiled_wiki_pages


def ingest_source(
    path: Path,
    llm: LLMClient,
    workspace: Path,
    settings: Settings | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> IngestResult:
    workspace = workspace.resolve()
    settings = settings or Settings.from_env(workspace)
    document = read_source(path, settings=settings)
    jsonl = JsonlStore(workspace)
    source_update_proposals = _source_update_proposals(document, jsonl, workspace)
    text_units = chunk_document(document)
    wiki_page = llm.generate_wiki_page(document)
    claims, evidence, entities, relations = llm.extract_knowledge(document, wiki_page)
    claims, entities, relations, governance_proposals = _apply_ingest_governance(
        claims=claims,
        entities=entities,
        relations=relations,
        evidence=evidence,
        workspace=workspace,
        settings=settings,
    )
    proposals = [*source_update_proposals, *governance_proposals]
    compiled_pages = build_compiled_wiki_pages(wiki_page, entities, claims, relations)

    markdown_store = MarkdownStore(workspace)
    markdown_store.save_page(wiki_page)
    for entity in entities:
        markdown_store.save_page(_entity_page(entity))
    for page in compiled_pages:
        markdown_store.save_page(page)
    markdown_store.update_index()
    markdown_store.append_log(
        f"ingested `{document.source_path}` as `{wiki_page.path}` with "
        f"{len(claims)} claims, {len(entities)} entities, {len(relations)} relations"
    )

    jsonl.upsert("documents.jsonl", [document])
    all_wiki_pages = [wiki_page, *(_entity_page(entity) for entity in entities), *compiled_pages]
    jsonl.upsert("wiki_pages.jsonl", all_wiki_pages)
    jsonl.upsert("claims.jsonl", claims)
    jsonl.upsert("evidence.jsonl", evidence)
    jsonl.upsert("nodes.jsonl", entities)
    jsonl.upsert("edges.jsonl", relations)
    if proposals:
        jsonl.upsert("proposals.jsonl", proposals)

    postgres = build_postgres_store(settings)
    if postgres:
        embedder = embedding_client or build_embedding_client(settings)
        wiki_pages = all_wiki_pages
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
        wiki_pages=all_wiki_pages,
        claims=claims,
        evidence=evidence,
        entities=entities,
        relations=relations,
        proposals=proposals,
    )


def _source_update_proposals(document: Document, jsonl: JsonlStore, workspace: Path) -> list:
    previous = [
        item
        for item in jsonl.load("documents.jsonl", Document)
        if item.source_path == document.source_path and item.hash != document.hash
    ]
    if not previous:
        return []
    old_source_ids = {item.id for item in previous}
    proposals = []
    for claim in jsonl.load("claims.jsonl", Claim):
        if old_source_ids.intersection(claim.source_ids):
            proposals.append(
                create_proposal(
                    "claim",
                    claim.id,
                    {
                        "title": f"Review outdated claim after source update: {claim.id}",
                        "rationale": f"Source path changed hash: {document.source_path}",
                        "status": "outdated",
                        "proposal_status": "pending_review",
                    },
                    workspace,
                    proposal_type="retire_claim",
                )
            )
    return proposals


def _apply_ingest_governance(
    claims: list[Claim],
    entities: list[Entity],
    relations: list[Relation],
    evidence: list[Evidence],
    workspace: Path,
    settings: Settings,
) -> tuple[list[Claim], list[Entity], list[Relation], list]:
    registry = build_ontology_registry(settings.ontology_profile)
    evidence_ids = {item.id for item in evidence}
    entity_ids = {item.id for item in entities}
    proposals = []

    governed_entities: list[Entity] = []
    for entity in entities:
        issues = registry.validate_entity(entity)
        if issues:
            entity.review_state = "pending_review"
            entity.governance_notes = "; ".join(issue.message for issue in issues)
            proposals.append(
                create_proposal(
                    "entity",
                    entity.id,
                    {
                        "title": f"Review entity ontology: {entity.name}",
                        "rationale": entity.governance_notes,
                        "entity_type": entity.entity_type,
                        "proposal_status": "pending_review",
                    },
                    workspace,
                    proposal_type="merge_entity",
                )
            )
        governed_entities.append(entity)

    governed_claims: list[Claim] = []
    for claim in claims:
        missing = sorted(set(claim.evidence_ids) - evidence_ids)
        if settings.governance_enforce_evidence and (not claim.evidence_ids or missing):
            proposals.append(
                create_proposal(
                    "claim",
                    claim.id,
                    {
                        "title": f"Claim needs evidence: {claim.id}",
                        "rationale": "Extracted claim did not include traceable evidence.",
                        "text": claim.text,
                        "evidence_ids": claim.evidence_ids,
                        "missing_evidence_ids": missing,
                        "proposal_status": "need_more_evidence",
                    },
                    workspace,
                    proposal_type="update_claim",
                )
            )
            continue
        governed_claims.append(claim)

    governed_relations: list[Relation] = []
    claim_ids = {claim.id for claim in governed_claims}
    for relation in relations:
        issues = registry.validate_relation(relation, entity_ids)
        missing_claims = sorted(set(relation.claim_ids) - claim_ids)
        missing_evidence = sorted(set(relation.evidence_ids) - evidence_ids)
        has_trace = bool(relation.claim_ids or relation.evidence_ids)
        hard_issues = [
            issue
            for issue in issues
            if issue.code in {"relation_missing_subject", "relation_missing_object"}
        ]
        if settings.governance_enforce_relation_trace and (not has_trace or missing_claims or missing_evidence or hard_issues):
            proposals.append(
                create_proposal(
                    "relation",
                    relation.id,
                    {
                        "title": f"Relation needs trace review: {relation.id}",
                        "rationale": "; ".join([issue.message for issue in issues] + [
                            f"Missing claims: {', '.join(missing_claims)}" if missing_claims else "",
                            f"Missing evidence: {', '.join(missing_evidence)}" if missing_evidence else "",
                            "Relation has no claim or evidence trace." if not has_trace else "",
                        ]).strip("; "),
                        "predicate": relation.predicate,
                        "claim_ids": relation.claim_ids,
                        "evidence_ids": relation.evidence_ids,
                        "proposal_status": "pending_review",
                    },
                    workspace,
                    proposal_type="update_relation",
                )
            )
            continue
        soft_issues = [issue for issue in issues if issue.code == "unknown_relation_predicate"]
        if soft_issues:
            relation.review_state = "pending_review"
            relation.governance_notes = "; ".join(issue.message for issue in soft_issues)
            proposals.append(
                create_proposal(
                    "relation",
                    relation.id,
                    {
                        "title": f"Review relation ontology: {relation.id}",
                        "rationale": relation.governance_notes,
                        "predicate": relation.predicate,
                        "proposal_status": "pending_review",
                    },
                    workspace,
                    proposal_type="update_relation",
                )
            )
        governed_relations.append(relation)
    return governed_claims, governed_entities, governed_relations, proposals


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
