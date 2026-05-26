from __future__ import annotations

import re

from llm_kg.models import Claim, Document, Entity, Evidence, Relation, WikiPage
from llm_kg.models.core import utc_now
from llm_kg.readers.common import slugify, stable_id


class MockLLMClient:
    """Deterministic local implementation used for tests and offline runs."""

    def generate_wiki_page(self, document: Document) -> WikiPage:
        summary = _summarize(document.content)
        key_points = _sentences(document.content)[:5]
        entity_names = _extract_entity_names(document.content)
        links = sorted({name for name in entity_names[:8]})
        link_text = ", ".join(f"[[{name}]]" for name in links) if links else "None"
        bullets = "\n".join(f"- {sentence}" for sentence in key_points) or "- No key points extracted."
        content = "\n".join(
            [
                f"# {document.title}",
                "",
                "## Metadata",
                f"- source_id: `{document.id}`",
                f"- source_path: `{document.source_path}`",
                f"- source_type: `{document.source_type}`",
                f"- hash: `{document.hash}`",
                "",
                "## Summary",
                summary,
                "",
                "## Key Claims",
                bullets,
                "",
                "## Related Entities",
                link_text,
                "",
            ]
        )
        path = f"wiki/sources/{slugify(document.title)}-{document.id[-8:]}.md"
        return WikiPage(
            id=stable_id("wiki", f"{document.id}:source"),
            title=document.title,
            page_type="source",
            path=path,
            content_md=content,
            source_ids=[document.id],
            wikilinks=links,
            tags=[document.source_type],
            updated_at=utc_now(),
        )

    def extract_knowledge(
        self, document: Document, wiki_page: WikiPage
    ) -> tuple[list[Claim], list[Evidence], list[Entity], list[Relation]]:
        sentences = _sentences(document.content)[:8]
        evidence: list[Evidence] = []
        claims: list[Claim] = []
        for sentence in sentences:
            evidence_id = stable_id("ev", f"{document.id}:{sentence}")
            claim_id = stable_id("claim", f"{document.id}:{sentence}")
            evidence.append(
                Evidence(
                    id=evidence_id,
                    source_id=document.id,
                    quote=sentence[:1000],
                    section=wiki_page.title,
                    confidence=0.72,
                )
            )
            subject, predicate, obj = _relation_parts(sentence)
            claims.append(
                Claim(
                    id=claim_id,
                    text=sentence,
                    source_ids=[document.id],
                    evidence_ids=[evidence_id],
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=0.68,
                    status="active",
                )
            )

        entity_names = _extract_entity_names(document.content)
        if not entity_names:
            entity_names = [document.title]
        entities = [
            Entity(
                id=stable_id("entity", name.lower()),
                name=name,
                entity_type=_entity_type(name),
                description=f"Entity extracted from {document.title}.",
                source_ids=[document.id],
            )
            for name in sorted(set(entity_names))[:20]
        ]
        entity_by_name = {entity.name.lower(): entity for entity in entities}
        relations: list[Relation] = []
        for claim in claims:
            if not claim.subject or not claim.object:
                continue
            subject = entity_by_name.get(claim.subject.lower())
            obj = entity_by_name.get(claim.object.lower())
            if not subject:
                subject = Entity(
                    id=stable_id("entity", claim.subject.lower()),
                    name=claim.subject,
                    entity_type=_entity_type(claim.subject),
                    source_ids=[document.id],
                )
                entities.append(subject)
                entity_by_name[subject.name.lower()] = subject
            if not obj:
                obj = Entity(
                    id=stable_id("entity", claim.object.lower()),
                    name=claim.object,
                    entity_type=_entity_type(claim.object),
                    source_ids=[document.id],
                )
                entities.append(obj)
                entity_by_name[obj.name.lower()] = obj
            relations.append(
                Relation(
                    id=stable_id("rel", f"{subject.id}:{claim.predicate}:{obj.id}"),
                    subject_id=subject.id,
                    predicate=claim.predicate or "mentions",
                    object_id=obj.id,
                    claim_ids=[claim.id],
                    evidence_ids=claim.evidence_ids,
                    confidence=claim.confidence,
                )
            )
        return claims, evidence, _dedupe_entities(entities), relations

    def answer_question(self, question: str, context: str) -> str:
        if not context.strip():
            return "No matching wiki pages, claims, or evidence were found."
        return (
            "Mock answer based on retrieved evidence. Review the cited hits below; "
            "no unsupported reasoning was generated in mock mode."
        )


def _sentences(content: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", content).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", normalized)
    return [part.strip() for part in parts if len(part.strip()) > 12]


def _summarize(content: str) -> str:
    sentences = _sentences(content)
    if not sentences:
        return "No summary available."
    return " ".join(sentences[:3])


def _extract_entity_names(content: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4}\b", content)
    ignored = {"The", "This", "That", "These", "Those", "A", "An", "I"}
    return [candidate.strip() for candidate in candidates if candidate.strip() not in ignored]


def _relation_parts(sentence: str) -> tuple[str | None, str | None, str | None]:
    patterns = [
        (r"(.+?)\s+(affects|supports|opposes|constrains|enables|mentions|updates)\s+(.+)", None),
        (r"(.+?)\s+(is|are|was|were)\s+(.+)", "is"),
        (r"(.+?)\s+(requires|required|limits|limited)\s+(.+)", None),
    ]
    for pattern, override in patterns:
        match = re.match(pattern, sentence, flags=re.IGNORECASE)
        if match:
            subject = _clean_entity_fragment(match.group(1))
            predicate = override or match.group(2).lower()
            obj = _clean_entity_fragment(match.group(3))
            if subject and obj:
                return subject, predicate.replace(" ", "_"), obj
    names = _extract_entity_names(sentence)
    if len(names) >= 2:
        return names[0], "mentions", names[1]
    return None, None, None


def _clean_entity_fragment(value: str) -> str:
    value = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", value)
    words = value.split()
    return " ".join(words[:6]).strip()


def _entity_type(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("policy", "law", "act", "sb ", "ab ")):
        return "policy"
    if any(token in lowered for token in ("city", "county", "department")):
        return "jurisdiction"
    if any(token in lowered for token in ("project", "development")):
        return "project"
    if any(token in lowered for token in ("risk", "issue")):
        return "risk"
    return "concept"


def _dedupe_entities(entities: list[Entity]) -> list[Entity]:
    deduped: dict[str, Entity] = {}
    for entity in entities:
        deduped[entity.id] = entity
    return list(deduped.values())
