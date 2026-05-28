from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime settings resolved from TOML config and environment variables."""

    workspace: Path = Field(default_factory=lambda: Path.cwd())
    llm_provider: str = "mock"
    openai_model: str = "gpt-4.1-mini"
    database_url: str | None = None
    embedding_provider: str = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    top_k: int = 5
    query_default_mode: str = "local"

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "Settings":
        resolved_workspace = workspace or Path(os.getenv("LLM_KG_WORKSPACE", Path.cwd()))
        file_values = _load_config_file(resolved_workspace)

        provider = os.getenv("LLM_KG_PROVIDER") or file_values.get("llm_provider")
        if provider is None:
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "mock"

        return cls(
            workspace=resolved_workspace,
            llm_provider=provider,
            database_url=os.getenv("LLM_KG_DATABASE_URL") or file_values.get("database_url"),
            openai_model=os.getenv("LLM_KG_OPENAI_MODEL")
            or file_values.get("openai_model")
            or "gpt-4.1-mini",
            embedding_provider=os.getenv("LLM_KG_EMBEDDING_PROVIDER")
            or file_values.get("embedding_provider")
            or provider,
            embedding_model=os.getenv("LLM_KG_EMBEDDING_MODEL")
            or file_values.get("embedding_model")
            or "text-embedding-3-small",
            embedding_dimensions=int(
                os.getenv("LLM_KG_EMBEDDING_DIMENSIONS") or file_values.get("embedding_dimensions") or "1536"
            ),
            top_k=int(os.getenv("LLM_KG_TOP_K") or file_values.get("top_k") or "5"),
            query_default_mode=os.getenv("LLM_KG_QUERY_MODE")
            or file_values.get("query_default_mode")
            or "local",
        )


def _load_config_file(workspace: Path) -> dict[str, object]:
    config_path = Path(os.getenv("LLM_KG_CONFIG", workspace / "llm_kg.toml"))
    if not config_path.exists():
        return {}

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    llm = raw.get("llm", {})
    database = raw.get("database", {})
    embedding = raw.get("embedding", {})
    query = raw.get("query", {})
    if not isinstance(llm, dict):
        raise ValueError("[llm] must be a TOML table")
    if not isinstance(database, dict):
        raise ValueError("[database] must be a TOML table")
    if not isinstance(embedding, dict):
        raise ValueError("[embedding] must be a TOML table")
    if not isinstance(query, dict):
        raise ValueError("[query] must be a TOML table")

    values: dict[str, object] = {}
    if "provider" in llm:
        values["llm_provider"] = llm["provider"]
    if "openai_model" in llm:
        values["openai_model"] = llm["openai_model"]
    if "url" in database:
        values["database_url"] = database["url"]
    if "provider" in embedding:
        values["embedding_provider"] = embedding["provider"]
    if "model" in embedding:
        values["embedding_model"] = embedding["model"]
    if "dimensions" in embedding:
        values["embedding_dimensions"] = embedding["dimensions"]
    if "top_k" in query:
        values["top_k"] = query["top_k"]
    if "default_mode" in query:
        values["query_default_mode"] = query["default_mode"]
    return values
