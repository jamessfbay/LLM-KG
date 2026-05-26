from __future__ import annotations

from pathlib import Path

from llm_kg.models import Document
from llm_kg.readers.common import build_document


def read_docx(path: Path) -> Document:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError("Reading .docx requires the python-docx package.") from exc
    doc = DocxDocument(str(path))
    content = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    return build_document(path, "docx", content)
