from pathlib import Path

import pytest

from llm_kg.llm.extraction_models import KnowledgeExtractionOutput
from llm_kg.llm.openai_client import OpenAILLMClient
from llm_kg.models import Document, WikiPage
from llm_kg.models.core import utc_now


class _Responses:
    def __init__(self, parsed=None, error: Exception | None = None) -> None:
        self.parsed = parsed
        self.error = error
        self.parse_called = False

    def parse(self, **kwargs):
        self.parse_called = True
        assert kwargs["text_format"] is KnowledgeExtractionOutput
        if self.error:
            raise self.error
        return type("Response", (), {"output_parsed": self.parsed})()

    def create(self, **kwargs):
        return type("Response", (), {"output_text": "# Wiki"})()


class _FakeClient:
    def __init__(self, responses: _Responses) -> None:
        self.responses = responses


def test_openai_extraction_uses_responses_parse(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    parsed = KnowledgeExtractionOutput.model_validate(
        {
            "evidence": [
                {
                    "id": "ev_project",
                    "source_id": "doc_1",
                    "quote": "3980 El Camino Real proposes housing.",
                    "page_number": 2,
                    "section": "A0.1",
                    "source_mode": "native_text",
                    "confidence": 0.9,
                }
            ],
            "claims": [
                {
                    "id": "claim_project",
                    "text": "3980 El Camino Real proposes housing.",
                    "source_ids": ["doc_1"],
                    "evidence_ids": ["ev_project"],
                    "subject": "3980 El Camino Real",
                    "predicate": "proposes",
                    "object": "housing",
                    "confidence": 0.88,
                    "rationale": "The quote states the proposed use.",
                }
            ],
            "entities": [
                {
                    "id": "entity_project",
                    "name": "3980 El Camino Real",
                    "entity_type": "project",
                    "aliases": [],
                    "description": "Planning project.",
                    "source_ids": ["doc_1"],
                }
            ],
            "relations": [],
        }
    )
    responses = _Responses(parsed=parsed)
    client = OpenAILLMClient(model="gpt-4.1-mini", workspace=tmp_path)
    client.client = _FakeClient(responses)

    claims, evidence, entities, relations = client.extract_knowledge(_document(), _wiki())

    assert responses.parse_called
    assert claims[0].id == "claim_project"
    assert evidence[0].page_number == 2
    assert evidence[0].source_mode == "native_text"
    assert entities[0].name == "3980 El Camino Real"
    assert relations == []


def test_openai_extraction_error_does_not_silently_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    responses = _Responses(error=ValueError("bad schema"))
    client = OpenAILLMClient(model="gpt-4.1-mini", workspace=tmp_path)
    client.client = _FakeClient(responses)

    with pytest.raises(RuntimeError, match="structured extraction failed"):
        client.extract_knowledge(_document(), _wiki())


def test_openai_extraction_can_fallback_when_explicit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    responses = _Responses(error=ValueError("bad schema"))
    client = OpenAILLMClient(model="gpt-4.1-mini", workspace=tmp_path, fallback_to_mock=True)
    client.client = _FakeClient(responses)

    claims, evidence, entities, _relations = client.extract_knowledge(_document(), _wiki())

    assert claims
    assert evidence
    assert entities


def _document() -> Document:
    return Document(
        id="doc_1",
        title="3980 El Camino Real",
        source_path="/tmp/3980.pdf",
        source_type="pdf",
        content="[Page 2 | native_text]\n3980 El Camino Real proposes housing.",
        hash="abc",
        metadata={"pdf_page_coverage": {"total_pages": 2, "text_pages": 2}},
    )


def _wiki() -> WikiPage:
    return WikiPage(
        id="wiki_1",
        title="3980 El Camino Real",
        page_type="source",
        path="wiki/sources/3980.md",
        content_md="# 3980 El Camino Real",
        source_ids=["doc_1"],
        updated_at=utc_now(),
    )
