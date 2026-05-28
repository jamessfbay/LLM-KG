from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.embeddings import build_embedding_client
from llm_kg.governance import apply_update_plan as _apply_update_plan
from llm_kg.governance import create_proposal as _create_proposal
from llm_kg.governance import export_proposal as _export_proposal
from llm_kg.governance import trace_object as _trace_object
from llm_kg.governance import verify_claim as _verify_claim
from llm_kg.llm import LLMClient, build_llm_client
from llm_kg.models import ApplyResult, IngestResult, LintIssue, QueryResult, TraceResult, UpdateProposalDraft, VerificationResult
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


def verify_claim(claim_id: str, workspace: Path | None = None) -> VerificationResult:
    settings = Settings.from_env(workspace)
    return _verify_claim(claim_id, settings.workspace)


def trace_object(target_type: str, target_id: str, workspace: Path | None = None) -> TraceResult:
    settings = Settings.from_env(workspace)
    return _trace_object(target_type, target_id, settings.workspace)


def create_proposal(
    target_type: str,
    target_id: str,
    proposed_change: dict,
    workspace: Path | None = None,
) -> UpdateProposalDraft:
    settings = Settings.from_env(workspace)
    return _create_proposal(target_type, target_id, proposed_change, settings.workspace)


def export_proposal(proposal_id: str, workspace: Path | None = None, export_format: str = "llm-kee") -> dict:
    settings = Settings.from_env(workspace)
    return _export_proposal(proposal_id, settings.workspace, export_format=export_format)


def apply_update_plan(plan: dict, workspace: Path | None = None) -> ApplyResult:
    settings = Settings.from_env(workspace)
    return _apply_update_plan(plan, settings.workspace)
