from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    model: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...
