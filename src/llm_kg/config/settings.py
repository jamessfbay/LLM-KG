from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime settings resolved from environment variables."""

    workspace: Path = Field(default_factory=lambda: Path.cwd())
    llm_provider: str = "mock"
    openai_model: str = "gpt-4.1-mini"
    top_k: int = 5

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "Settings":
        provider = os.getenv("LLM_KG_PROVIDER")
        if provider is None:
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "mock"
        return cls(
            workspace=workspace or Path(os.getenv("LLM_KG_WORKSPACE", Path.cwd())),
            llm_provider=provider,
            openai_model=os.getenv("LLM_KG_OPENAI_MODEL", "gpt-4.1-mini"),
            top_k=int(os.getenv("LLM_KG_TOP_K", "5")),
        )
