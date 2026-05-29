from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

from llm_kg.llm.extraction_models import KnowledgeExtractionOutput
from llm_kg.llm.mock_client import MockLLMClient
from llm_kg.models import Claim, Document, Entity, Evidence, Relation, WikiPage
from llm_kg.prompts import PromptLoader
from llm_kg.readers.common import stable_id


class OpenAILLMClient:
    """OpenAI-backed client using Structured Outputs for extraction."""

    def __init__(self, model: str, workspace: Path | None = None, fallback_to_mock: bool = False) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI provider requires the openai package.") from exc
        self.model = model
        self.client = OpenAI()
        self.fallback_to_mock = fallback_to_mock
        self.fallback = MockLLMClient()
        self.prompts = PromptLoader(workspace) if workspace else None

    def generate_wiki_page(self, document: Document) -> WikiPage:
        instructions = _prompt(self.prompts, "generate_wiki_page")
        prompt = (
            f"{instructions}\n\n"
            f"Title: {document.title}\nSource ID: {document.id}\nContent:\n{document.content[:16000]}"
        )
        text = self._complete(prompt)
        page = self.fallback.generate_wiki_page(document)
        if text.strip():
            page.content_md = text
            coverage = document.metadata.get("pdf_page_coverage") if document.metadata else None
            if isinstance(coverage, dict):
                page.content_md = page.content_md.rstrip() + "\n\n## Extraction Coverage\n" + "\n".join(
                    [
                        f"- total_pages: `{coverage.get('total_pages', 0)}`",
                        f"- native_pages: `{coverage.get('native_pages', 0)}`",
                        f"- ocr_pages: `{coverage.get('ocr_pages', 0)}`",
                        f"- timeout_pages: `{coverage.get('timeout_pages', 0)}`",
                        f"- failed_pages: `{coverage.get('failed_pages', 0)}`",
                    ]
                )
        return page

    def extract_knowledge(
        self, document: Document, wiki_page: WikiPage
    ) -> tuple[list[Claim], list[Evidence], list[Entity], list[Relation]]:
        instructions = "\n\n".join(
            [
                _prompt(self.prompts, "extract_claims"),
                _prompt(self.prompts, "extract_entities"),
                _prompt(self.prompts, "extract_relations"),
            ]
        )
        prompt = (
            f"{instructions}\n\n"
            "Return records that match the provided structured output schema. "
            "Every claim must reference existing evidence_ids. Every evidence record must quote source text exactly. "
            "Use page_number and source_mode when available from [Page N | mode] markers.\n\n"
            f"Document ID: {document.id}\n"
            f"Source path: {document.source_path}\n"
            f"Page coverage: {document.metadata.get('pdf_page_coverage', {})}\n\n"
            f"Page-aware source text:\n{_source_window(document.content)}\n\n"
            f"Generated wiki context:\n{wiki_page.content_md[:6000]}"
        )
        try:
            parsed = self._parse_extraction(prompt)
            return _to_core_records(parsed, document)
        except Exception as exc:
            if self.fallback_to_mock or _env_true("LLM_KG_OPENAI_FALLBACK_TO_MOCK"):
                return self.fallback.extract_knowledge(document, wiki_page)
            raise RuntimeError(
                "OpenAI structured extraction failed. Set LLM_KG_OPENAI_FALLBACK_TO_MOCK=true "
                "to allow deterministic mock fallback."
            ) from exc

    def answer_question(self, question: str, context: str, mode: str = "local") -> str:
        instructions = _prompt(self.prompts, "answer_basic" if mode == "basic" else "answer_local")
        prompt = (
            f"{instructions}\n\n"
            f"Question: {question}\n\nContext:\n{context[:16000]}"
        )
        return self._complete(prompt).strip() or self.fallback.answer_question(question, context, mode=mode)

    def _complete(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text

    def _parse_extraction(self, prompt: str) -> KnowledgeExtractionOutput:
        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=KnowledgeExtractionOutput,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("OpenAI response did not include parsed structured output.")
        if isinstance(parsed, KnowledgeExtractionOutput):
            return parsed
        try:
            return KnowledgeExtractionOutput.model_validate(parsed)
        except ValidationError as exc:
            raise ValueError(f"OpenAI structured extraction did not match schema: {exc}") from exc


def _prompt(loader: PromptLoader | None, name: str) -> str:
    if loader is None:
        return ""
    try:
        return loader.load(name)
    except FileNotFoundError:
        return ""


def _source_window(content: str, max_chars: int = 24000) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Source text truncated for extraction input.]"


def _to_core_records(
    parsed: KnowledgeExtractionOutput,
    document: Document,
) -> tuple[list[Claim], list[Evidence], list[Entity], list[Relation]]:
    evidence = [
        Evidence(
            id=item.id or stable_id("ev", f"{document.id}:{item.quote}"),
            source_id=item.source_id or document.id,
            quote=item.quote,
            page_number=item.page_number,
            section=item.section,
            source_mode=item.source_mode,
            confidence=item.confidence,
        )
        for item in parsed.evidence
        if item.quote.strip()
    ]
    entities = [
        Entity(
            id=item.id or stable_id("entity", item.name.lower()),
            name=item.name,
            entity_type=item.entity_type,
            aliases=item.aliases,
            description=item.description,
            source_ids=item.source_ids or [document.id],
        )
        for item in parsed.entities
        if item.name.strip()
    ]
    claims = [
        Claim(
            id=item.id or stable_id("claim", f"{document.id}:{item.text}"),
            text=item.text,
            source_ids=item.source_ids or [document.id],
            evidence_ids=item.evidence_ids,
            subject=item.subject,
            predicate=item.predicate,
            object=item.object,
            confidence=item.confidence,
            governance_notes=item.rationale,
        )
        for item in parsed.claims
        if item.text.strip()
    ]
    relations = [
        Relation(
            id=item.id or stable_id("rel", f"{item.subject_id}:{item.predicate}:{item.object_id}"),
            subject_id=item.subject_id,
            predicate=item.predicate,
            object_id=item.object_id,
            claim_ids=item.claim_ids,
            evidence_ids=item.evidence_ids,
            confidence=item.confidence,
            governance_notes=item.rationale,
        )
        for item in parsed.relations
        if item.subject_id and item.object_id and item.predicate
    ]
    return claims, evidence, entities, relations


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
