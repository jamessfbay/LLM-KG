from __future__ import annotations

from collections import Counter, defaultdict

from llm_kg.models import Claim, Entity, Relation, WikiPage
from llm_kg.models.core import utc_now
from llm_kg.readers.common import slugify, stable_id


def build_compiled_wiki_pages(
    source_page: WikiPage,
    entities: list[Entity],
    claims: list[Claim],
    relations: list[Relation],
) -> list[WikiPage]:
    pages: list[WikiPage] = []
    pages.extend(_concept_pages(source_page, entities, claims))
    synthesis = _synthesis_page(source_page, entities, claims, relations)
    if synthesis:
        pages.append(synthesis)
    comparison = _comparison_page(source_page, entities, claims)
    if comparison:
        pages.append(comparison)
    return pages


def _concept_pages(source_page: WikiPage, entities: list[Entity], claims: list[Claim]) -> list[WikiPage]:
    grouped: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.entity_type].append(entity)

    pages: list[WikiPage] = []
    for entity_type, items in sorted(grouped.items()):
        if entity_type in {"person", "organization", "project", "document"}:
            continue
        related_claims = [
            claim for claim in claims if claim.subject in {item.name for item in items} or claim.object in {item.name for item in items}
        ][:10]
        lines = [
            f"# {entity_type.title()} Concepts",
            "",
            "## Entities",
            *(f"- [[{entity.name}]]" for entity in sorted(items, key=lambda item: item.name.lower())[:30]),
            "",
            "## Evidence-backed Claims",
            *(f"- {claim.text} (`{', '.join(claim.evidence_ids)}`)" for claim in related_claims),
            "",
        ]
        page_id = stable_id("wiki", f"concept:{source_page.id}:{entity_type}")
        pages.append(
            WikiPage(
                id=page_id,
                title=f"{entity_type.title()} Concepts",
                page_type="concept",
                path=f"wiki/concepts/{slugify(entity_type)}-{page_id[-8:]}.md",
                content_md="\n".join(lines),
                source_ids=source_page.source_ids,
                wikilinks=[entity.name for entity in items],
                tags=[entity_type],
                updated_at=utc_now(),
            )
        )
    return pages


def _synthesis_page(
    source_page: WikiPage,
    entities: list[Entity],
    claims: list[Claim],
    relations: list[Relation],
) -> WikiPage | None:
    if not claims and not relations:
        return None
    entity_counts = Counter(entity.entity_type for entity in entities)
    lines = [
        f"# Synthesis: {source_page.title}",
        "",
        "## Structure",
        *(f"- {entity_type}: {count}" for entity_type, count in sorted(entity_counts.items())),
        "",
        "## Key Claims",
        *(f"- {claim.text} (`{', '.join(claim.evidence_ids)}`)" for claim in claims[:12]),
        "",
        "## Relation Paths",
        *(f"- `{relation.subject_id}` -> `{relation.predicate}` -> `{relation.object_id}`" for relation in relations[:12]),
        "",
    ]
    page_id = stable_id("wiki", f"synthesis:{source_page.id}")
    return WikiPage(
        id=page_id,
        title=f"Synthesis: {source_page.title}",
        page_type="synthesis",
        path=f"wiki/synthesis/{slugify(source_page.title)}-{page_id[-8:]}.md",
        content_md="\n".join(lines),
        source_ids=source_page.source_ids,
        wikilinks=[entity.name for entity in entities[:20]],
        tags=["synthesis"],
        updated_at=utc_now(),
    )


def _comparison_page(source_page: WikiPage, entities: list[Entity], claims: list[Claim]) -> WikiPage | None:
    comparable = [entity for entity in entities if entity.entity_type in {"project", "policy", "organization", "document"}]
    if len(comparable) < 2:
        return None
    lines = [
        f"# Comparison: {source_page.title}",
        "",
        "| Entity | Type | Evidence-backed Claims |",
        "| --- | --- | --- |",
    ]
    for entity in comparable[:12]:
        count = sum(1 for claim in claims if claim.subject == entity.name or claim.object == entity.name)
        lines.append(f"| [[{entity.name}]] | {entity.entity_type} | {count} |")
    lines.append("")
    page_id = stable_id("wiki", f"comparison:{source_page.id}")
    return WikiPage(
        id=page_id,
        title=f"Comparison: {source_page.title}",
        page_type="comparison",
        path=f"wiki/comparisons/{slugify(source_page.title)}-{page_id[-8:]}.md",
        content_md="\n".join(lines),
        source_ids=source_page.source_ids,
        wikilinks=[entity.name for entity in comparable[:12]],
        tags=["comparison"],
        updated_at=utc_now(),
    )
