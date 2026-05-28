from __future__ import annotations

import json
from pathlib import Path

from llm_kg.llm.mock_client import MockLLMClient
from llm_kg.models import Claim, Document, Entity, Evidence, Relation, WikiPage
from llm_kg.prompts import PromptLoader


class OpenAILLMClient:
    """OpenAI-backed client with deterministic mock fallback for malformed outputs."""

    def __init__(self, model: str, workspace: Path | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI provider requires the openai package.") from exc
        self.model = model
        self.client = OpenAI()
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
            "Return one strict JSON object with keys: claims, evidence, entities, relations.\n\n"
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


def _prompt(loader: PromptLoader | None, name: str) -> str:
    if loader is None:
        return ""
    try:
        return loader.load(name)
    except FileNotFoundError:
        return ""
