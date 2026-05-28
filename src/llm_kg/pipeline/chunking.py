from __future__ import annotations

import re

from llm_kg.models import Document, TextUnit
from llm_kg.readers.common import stable_id


def chunk_document(document: Document, max_chars: int = 1800, overlap_chars: int = 180) -> list[TextUnit]:
    """Create deterministic text units with simple character windows."""

    content = re.sub(r"\s+", " ", document.content).strip()
    if not content:
        return []
    chunks: list[TextUnit] = []
    start = 0
    index = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            boundary = content.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        text = content[start:end].strip()
        if text:
            chunks.append(
                TextUnit(
                    id=stable_id("tu", f"{document.id}:{index}:{start}:{end}:{text}"),
                    document_id=document.id,
                    text=text,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    token_count=len(re.findall(r"\S+", text)),
                    source_ids=[document.id],
                )
            )
            index += 1
        if end >= len(content):
            break
        start = max(end - overlap_chars, 0)
    return chunks
