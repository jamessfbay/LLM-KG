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
    top_k: int = 5

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
            openai_model=os.getenv("LLM_KG_OPENAI_MODEL")
            or file_values.get("openai_model")
            or "gpt-4.1-mini",
            top_k=int(os.getenv("LLM_KG_TOP_K") or file_values.get("top_k") or "5"),
        )


def _load_config_file(workspace: Path) -> dict[str, object]:
    config_path = Path(os.getenv("LLM_KG_CONFIG", workspace / "llm_kg.toml"))
    if not config_path.exists():
        return {}

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    llm = raw.get("llm", {})
    query = raw.get("query", {})
    if not isinstance(llm, dict):
        raise ValueError("[llm] must be a TOML table")
    if not isinstance(query, dict):
        raise ValueError("[query] must be a TOML table")

    values: dict[str, object] = {}
    if "provider" in llm:
        values["llm_provider"] = llm["provider"]
    if "openai_model" in llm:
        values["openai_model"] = llm["openai_model"]
    if "top_k" in query:
        values["top_k"] = query["top_k"]
    return values
