from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.governance import apply_kee_plan as _apply_kee_plan
from llm_kg.embeddings import build_embedding_client
from llm_kg.governance import apply_update_plan as _apply_update_plan
from llm_kg.governance import create_proposal as _create_proposal
from llm_kg.governance import export_proposal as _export_proposal
from llm_kg.governance import import_kee_decision as _import_kee_decision
from llm_kg.governance import trace_object as _trace_object
from llm_kg.governance import verify_claim as _verify_claim
from llm_kg.governance import verify_object as _verify_object
from llm_kg.llm import LLMClient, build_llm_client
from llm_kg.models import ApplyResult, IngestResult, LintIssue, QueryResult, ReasoningTrace, TraceResult, UpdateProposalDraft, VerificationResult
from llm_kg.pipeline.ingest import ingest_source as _ingest_source
from llm_kg.pipeline.lint import lint_workspace as _lint_workspace
from llm_kg.reasoning.answer_question import query_knowledge as _query_knowledge
from llm_kg.reasoning.traces import export_reasoning_trace as _export_reasoning_trace
from llm_kg.reasoning.traces import get_reasoning_trace as _get_reasoning_trace
from llm_kg.reasoning.traces import list_reasoning_traces as _list_reasoning_traces
from llm_kg.validation import cross_validate_claims as _cross_validate_claims


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
    persist_trace: bool = True,
) -> QueryResult:
    settings = Settings.from_env(workspace)
    client = build_llm_client(settings)
    return _query_knowledge(
        question,
        client,
        settings.workspace,
        top_k=top_k,
        mode=mode or settings.query_default_mode,
        persist_trace=persist_trace,
    )


def lint_workspace(workspace: Path | None = None) -> list[LintIssue]:
    settings = Settings.from_env(workspace)
    return _lint_workspace(settings.workspace)


def verify_claim(claim_id: str, workspace: Path | None = None) -> VerificationResult:
    settings = Settings.from_env(workspace)
    return _verify_claim(claim_id, settings.workspace)


def verify_object(target_type: str, target_id: str, workspace: Path | None = None) -> VerificationResult:
    settings = Settings.from_env(workspace)
    return _verify_object(target_type, target_id, settings.workspace)


def trace_object(target_type: str, target_id: str, workspace: Path | None = None) -> TraceResult:
    settings = Settings.from_env(workspace)
    return _trace_object(target_type, target_id, settings.workspace)


def trace_query(trace_id: str, workspace: Path | None = None) -> ReasoningTrace:
    settings = Settings.from_env(workspace)
    trace = _get_reasoning_trace(trace_id, settings.workspace)
    if not trace:
        raise ValueError(f"Reasoning trace not found: {trace_id}")
    return trace


def list_reasoning_traces(workspace: Path | None = None) -> list[ReasoningTrace]:
    settings = Settings.from_env(workspace)
    return _list_reasoning_traces(settings.workspace)


def export_reasoning_trace(trace_id: str, workspace: Path | None = None, export_format: str = "llm-kee") -> dict:
    settings = Settings.from_env(workspace)
    return _export_reasoning_trace(trace_id, settings.workspace, export_format=export_format)


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


def apply_kee_plan(plan: dict, workspace: Path | None = None) -> ApplyResult:
    settings = Settings.from_env(workspace)
    return _apply_kee_plan(plan, settings.workspace)


def import_kee_decision(decision: dict, workspace: Path | None = None) -> dict:
    settings = Settings.from_env(workspace)
    return _import_kee_decision(decision, settings.workspace)


def cross_validate_claims(
    workspace: Path | None = None,
    providers: list[str] | None = None,
    limit: int | None = None,
    output_path: Path | None = None,
):
    settings = Settings.from_env(workspace)
    return _cross_validate_claims(
        settings.workspace,
        providers=providers,
        limit=limit,
        output_path=output_path,
        settings=settings,
    )
