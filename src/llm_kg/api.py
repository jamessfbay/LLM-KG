from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.embeddings import build_embedding_client
from llm_kg.llm import LLMClient, build_llm_client
from llm_kg.models import IngestResult, LintIssue, QueryResult
from llm_kg.pipeline.ingest import ingest_source as _ingest_source
from llm_kg.pipeline.lint import lint_workspace as _lint_workspace
from llm_kg.reasoning.answer_question import query_knowledge as _query_knowledge


def ingest_source(path: Path, llm: LLMClient | None = None, workspace: Path | None = None) -> IngestResult:
    settings = Settings.from_env(workspace)
    client = llm or build_llm_client(settings)
    embedding_client = build_embedding_client(settings) if settings.database_url else None
    return _ingest_source(Path(path), client, settings.workspace, settings=settings, embedding_client=embedding_client)


def query_knowledge(
    question: str,
    workspace: Path | None = None,
    top_k: int = 5,
    mode: str | None = None,
) -> QueryResult:
    settings = Settings.from_env(workspace)
    client = build_llm_client(settings)
    return _query_knowledge(question, client, settings.workspace, top_k=top_k, mode=mode or settings.query_default_mode)


def lint_workspace(workspace: Path | None = None) -> list[LintIssue]:
    settings = Settings.from_env(workspace)
    return _lint_workspace(settings.workspace)
