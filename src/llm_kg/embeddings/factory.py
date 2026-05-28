from __future__ import annotations

from llm_kg.config import Settings
from llm_kg.embeddings.mock_client import MockEmbeddingClient
from llm_kg.embeddings.openai_client import OpenAIEmbeddingClient
from llm_kg.embeddings.protocol import EmbeddingClient


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingClient(model=settings.embedding_model, dimensions=settings.embedding_dimensions)
    if settings.embedding_provider == "mock":
        return MockEmbeddingClient(model=settings.embedding_model, dimensions=settings.embedding_dimensions)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
