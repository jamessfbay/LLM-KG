from __future__ import annotations

import hashlib
import re
from pathlib import Path

from llm_kg.models import Document


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "untitled"


def build_document(path: Path, source_type: str, content: str) -> Document:
    if not content.strip():
        raise ValueError(f"Source is empty: {path}")
    digest = file_hash(path)
    return Document(
        id=stable_id("doc", digest),
        title=path.stem,
        source_path=str(path),
        source_type=source_type,  # type: ignore[arg-type]
        content=content,
        hash=digest,
    )
