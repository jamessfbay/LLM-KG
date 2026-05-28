from llm_kg.models import Document
from llm_kg.pipeline.chunking import chunk_document


def test_chunk_document_creates_stable_text_units() -> None:
    doc = Document(
        id="doc_1",
        title="Source",
        source_path="source.md",
        source_type="md",
        content="Alpha affects Beta. " * 80,
        hash="hash",
    )

    first = chunk_document(doc, max_chars=120, overlap_chars=20)
    second = chunk_document(doc, max_chars=120, overlap_chars=20)

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) > 1
    assert first[0].document_id == "doc_1"
    assert first[0].token_count > 0
