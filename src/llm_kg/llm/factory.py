from __future__ import annotations

from llm_kg.config import Settings
from llm_kg.llm.mock_client import MockLLMClient
from llm_kg.llm.openai_client import OpenAILLMClient
from llm_kg.llm.protocol import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        return OpenAILLMClient(
            model=settings.openai_model,
            workspace=settings.workspace,
            fallback_to_mock=settings.llm_fallback_to_mock,
        )
    if settings.llm_provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
