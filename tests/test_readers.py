from pathlib import Path

import pytest

from llm_kg.config import Settings
from llm_kg.readers import read_source
from llm_kg.readers import pdf_reader


def test_txt_reader_builds_stable_document(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("SB 330 affects Housing Project Alpha.", encoding="utf-8")

    first = read_source(path)
    second = read_source(path)

    assert first.id == second.id
    assert first.hash == second.hash
    assert first.source_type == "txt"
    assert "Housing Project Alpha" in first.content


def test_markdown_reader(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Note\n\nPalo Alto supports Project Beta.", encoding="utf-8")

    doc = read_source(path)

    assert doc.title == "note"
    assert doc.source_type == "md"


def test_empty_file_errors(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        read_source(path)


def test_unsupported_suffix_errors(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        read_source(path)


def test_docx_reader(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    from docx import Document as DocxDocument

    path = tmp_path / "source.docx"
    docx = DocxDocument()
    docx.add_paragraph("SB 330 limits City Discretion.")
    docx.save(path)

    doc = read_source(path)

    assert doc.source_type == "docx"
    assert "City Discretion" in doc.content


def test_pdf_reader_dependency_path(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    # PDF text extraction varies by producer. This test verifies dispatch and the
    # reader's clear empty-source behavior without adding a PDF generation dep.
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ValueError, match="empty"):
        read_source(path)


def test_pdf_reader_ocr_fallback_for_failed_page(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    path = tmp_path / "ocr.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)

    def fake_extract(reader, page_index):
        return pdf_reader.PageExtraction(
            page_number=page_index + 1,
            text="[Page 1 text extraction failed]",
            mode="failed_placeholder",
            error="forced",
        )

    class FakeOCR:
        def __init__(self, model: str, timeout: int) -> None:
            self.model = model
            self.timeout = timeout

        def extract_text(self, image_bytes: bytes, page_number: int) -> str:
            return "3980 El Camino Real plan set page text."

    monkeypatch.setattr(pdf_reader, "_extract_page_text_inline", fake_extract)
    monkeypatch.setattr(pdf_reader, "_render_pdf_page", lambda path, page_number: b"png")
    monkeypatch.setattr(pdf_reader, "_OpenAIVisionOCR", FakeOCR)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    settings = Settings(workspace=tmp_path, ocr_provider="openai", ocr_model="gpt-4.1-mini", ocr_max_pages=1)
    doc = pdf_reader.read_pdf(path, settings=settings)

    assert "[Page 1 | ocr_text | provider=openai]" in doc.content
    assert "3980 El Camino Real" in doc.content
    assert doc.metadata["pdf_page_coverage"]["ocr_pages"] == 1
    assert doc.metadata["pdf_page_coverage"]["failed_pages"] == 0


def test_pdf_reader_ocr_missing_key_keeps_clear_warning(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    path = tmp_path / "failed.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)

    def fake_extract(reader, page_index):
        return pdf_reader.PageExtraction(
            page_number=page_index + 1,
            text="[Page 1 text extraction failed]",
            mode="failed_placeholder",
            error="forced",
        )

    monkeypatch.setattr(pdf_reader, "_extract_page_text_inline", fake_extract)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings(workspace=tmp_path, ocr_provider="openai", ocr_model="gpt-4.1-mini", ocr_max_pages=1)
    with pytest.raises(ValueError, match="failed for all pages"):
        pdf_reader.read_pdf(path, settings=settings)
