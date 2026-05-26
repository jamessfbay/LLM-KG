from __future__ import annotations

from pathlib import Path

from llm_kg.models import Document
from llm_kg.readers.common import build_document


def read_pdf(path: Path) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading .pdf requires the pypdf package.") from exc
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return build_document(path, "pdf", "\n\n".join(page for page in pages if page.strip()))
