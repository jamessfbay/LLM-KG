from __future__ import annotations

from typing import Protocol

from llm_kg.models import Claim, Document, Entity, Evidence, Relation, WikiPage


class LLMClient(Protocol):
    def generate_wiki_page(self, document: Document) -> WikiPage:
        ...

    def extract_knowledge(
        self, document: Document, wiki_page: WikiPage
    ) -> tuple[list[Claim], list[Evidence], list[Entity], list[Relation]]:
        ...

    def answer_question(self, question: str, context: str) -> str:
        ...
