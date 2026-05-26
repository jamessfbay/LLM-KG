from __future__ import annotations

from pathlib import Path

from llm_kg.models import Document
from llm_kg.readers.docx_reader import read_docx
from llm_kg.readers.markdown_reader import read_markdown
from llm_kg.readers.pdf_reader import read_pdf
from llm_kg.readers.text_reader import read_text


def read_source(path: Path) -> Document:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise ValueError(f"Expected a file, got directory: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_text(path)
    if suffix == ".md":
        return read_markdown(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported source type '{suffix}'. Supported: .txt, .md, .docx, .pdf")
