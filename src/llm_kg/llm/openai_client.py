from __future__ import annotations

import json

from llm_kg.llm.mock_client import MockLLMClient
from llm_kg.models import Claim, Document, Entity, Evidence, Relation, WikiPage


class OpenAILLMClient:
    """OpenAI-backed client with deterministic mock fallback for malformed outputs."""

    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI provider requires the openai package.") from exc
        self.model = model
        self.client = OpenAI()
        self.fallback = MockLLMClient()

    def generate_wiki_page(self, document: Document) -> WikiPage:
        prompt = (
            "Generate a concise Markdown LLM wiki source page. Include Metadata, "
            "Summary, Key Claims, and Related Entities sections. Use wikilinks as [[Name]].\n\n"
            f"Title: {document.title}\nSource ID: {document.id}\nContent:\n{document.content[:16000]}"
        )
        text = self._complete(prompt)
        page = self.fallback.generate_wiki_page(document)
        if text.strip():
            page.content_md = text
        return page

    def extract_knowledge(
        self, document: Document, wiki_page: WikiPage
    ) -> tuple[list[Claim], list[Evidence], list[Entity], list[Relation]]:
        prompt = (
            "Extract knowledge as JSON with keys claims, evidence, entities, relations. "
            "Use IDs if obvious, otherwise omit and they will be generated later. "
            "Every claim must cite evidence_ids.\n\n"
            f"Document ID: {document.id}\nWiki:\n{wiki_page.content_md[:12000]}"
        )
        text = self._complete(prompt)
        try:
            data = json.loads(text)
            claims = [Claim.model_validate(item) for item in data.get("claims", [])]
            evidence = [Evidence.model_validate(item) for item in data.get("evidence", [])]
            entities = [Entity.model_validate(item) for item in data.get("entities", [])]
            relations = [Relation.model_validate(item) for item in data.get("relations", [])]
            if claims or evidence or entities or relations:
                return claims, evidence, entities, relations
        except Exception:
            pass
        return self.fallback.extract_knowledge(document, wiki_page)

    def answer_question(self, question: str, context: str) -> str:
        prompt = (
            "Answer the question using only the supplied context. If evidence is insufficient, say so. "
            "Cite hit IDs where useful.\n\n"
            f"Question: {question}\n\nContext:\n{context[:16000]}"
        )
        return self._complete(prompt).strip() or self.fallback.answer_question(question, context)

    def _complete(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text
