from __future__ import annotations

from pathlib import Path

from llm_kg.models import Document
from llm_kg.readers.common import build_document


def read_text(path: Path) -> Document:
    return build_document(path, "txt", path.read_text(encoding="utf-8"))
