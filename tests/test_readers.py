from pathlib import Path

import pytest

from llm_kg.readers import read_source


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
