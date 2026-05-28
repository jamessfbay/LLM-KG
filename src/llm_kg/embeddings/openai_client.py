from __future__ import annotations


class OpenAIEmbeddingClient:
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI embeddings require the openai package.") from exc
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
        return [item.embedding for item in response.data]
