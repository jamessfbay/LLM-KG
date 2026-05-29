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
    llm_fallback_to_mock: bool = False
    database_url: str | None = None
    embedding_provider: str = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    top_k: int = 5
    query_default_mode: str = "local"
    ontology_profile: str = "generic"
    governance_enforce_evidence: bool = True
    governance_enforce_relation_trace: bool = True
    kee_workspace: Path | None = None
    kee_enable_direct_adapter: bool = True
    ocr_provider: str = "none"
    ocr_model: str | None = None
    ocr_max_pages: int = 25
    ocr_timeout_seconds: int = 30

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
            llm_fallback_to_mock=_bool(
                os.getenv("LLM_KG_OPENAI_FALLBACK_TO_MOCK"),
                file_values.get("llm_fallback_to_mock"),
                False,
            ),
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
            ontology_profile=str(os.getenv("LLM_KG_ONTOLOGY_PROFILE") or file_values.get("ontology_profile") or "generic"),
            governance_enforce_evidence=_bool(
                os.getenv("LLM_KG_ENFORCE_EVIDENCE"), file_values.get("governance_enforce_evidence"), True
            ),
            governance_enforce_relation_trace=_bool(
                os.getenv("LLM_KG_ENFORCE_RELATION_TRACE"),
                file_values.get("governance_enforce_relation_trace"),
                True,
            ),
            kee_workspace=_optional_path(os.getenv("LLM_KG_KEE_WORKSPACE") or file_values.get("kee_workspace")),
            kee_enable_direct_adapter=_bool(
                os.getenv("LLM_KG_KEE_ENABLE_DIRECT_ADAPTER"),
                file_values.get("kee_enable_direct_adapter"),
                True,
            ),
            ocr_provider=str(os.getenv("LLM_KG_OCR_PROVIDER") or file_values.get("ocr_provider") or "none"),
            ocr_model=str(
                os.getenv("LLM_KG_OCR_MODEL")
                or file_values.get("ocr_model")
                or os.getenv("LLM_KG_OPENAI_MODEL")
                or file_values.get("openai_model")
                or "gpt-4.1-mini"
            ),
            ocr_max_pages=int(os.getenv("LLM_KG_OCR_MAX_PAGES") or file_values.get("ocr_max_pages") or "25"),
            ocr_timeout_seconds=int(
                os.getenv("LLM_KG_OCR_TIMEOUT_SECONDS") or file_values.get("ocr_timeout_seconds") or "30"
            ),
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
    ontology = raw.get("ontology", {})
    governance = raw.get("governance", {})
    kee = raw.get("kee", {})
    ocr = raw.get("ocr", {})
    if not isinstance(llm, dict):
        raise ValueError("[llm] must be a TOML table")
    if not isinstance(database, dict):
        raise ValueError("[database] must be a TOML table")
    if not isinstance(embedding, dict):
        raise ValueError("[embedding] must be a TOML table")
    if not isinstance(query, dict):
        raise ValueError("[query] must be a TOML table")
    if not isinstance(ontology, dict):
        raise ValueError("[ontology] must be a TOML table")
    if not isinstance(governance, dict):
        raise ValueError("[governance] must be a TOML table")
    if not isinstance(kee, dict):
        raise ValueError("[kee] must be a TOML table")
    if not isinstance(ocr, dict):
        raise ValueError("[ocr] must be a TOML table")

    values: dict[str, object] = {}
    if "provider" in llm:
        values["llm_provider"] = llm["provider"]
    if "openai_model" in llm:
        values["openai_model"] = llm["openai_model"]
    if "fallback_to_mock" in llm:
        values["llm_fallback_to_mock"] = llm["fallback_to_mock"]
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
    if "profile" in ontology:
        values["ontology_profile"] = ontology["profile"]
    if "enforce_evidence" in governance:
        values["governance_enforce_evidence"] = governance["enforce_evidence"]
    if "enforce_relation_trace" in governance:
        values["governance_enforce_relation_trace"] = governance["enforce_relation_trace"]
    if "workspace" in kee:
        values["kee_workspace"] = kee["workspace"]
    if "enable_direct_adapter" in kee:
        values["kee_enable_direct_adapter"] = kee["enable_direct_adapter"]
    if "provider" in ocr:
        values["ocr_provider"] = ocr["provider"]
    if "model" in ocr:
        values["ocr_model"] = ocr["model"]
    if "max_pages" in ocr:
        values["ocr_max_pages"] = ocr["max_pages"]
    if "timeout_seconds" in ocr:
        values["ocr_timeout_seconds"] = ocr["timeout_seconds"]
    return values


def _bool(env_value: str | None, file_value: object, default: bool) -> bool:
    value = env_value if env_value is not None else file_value
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_path(value: object) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value)).expanduser()
