from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ReviewState = Literal["auto_accepted", "pending_review", "approved", "rejected", "superseded"]


class GovernanceFields(BaseModel):
    review_state: ReviewState = "auto_accepted"
    version: int = 1
    created_by: str = "system"
    updated_by: str = "system"
    updated_at: datetime = Field(default_factory=utc_now)
    supersedes_id: str | None = None
    superseded_by_id: str | None = None
    governance_notes: str | None = None


class Document(BaseModel):
    id: str
    title: str
    source_path: str
    source_type: Literal["txt", "md", "docx", "pdf"]
    content: str
    author: str | None = None
    created_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=utc_now)
    hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WikiPage(GovernanceFields):
    id: str
    title: str
    page_type: Literal["source", "entity", "concept", "synthesis", "comparison"]
    path: str
    content_md: str
    source_ids: list[str] = Field(default_factory=list)
    wikilinks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class TextUnit(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    source_ids: list[str] = Field(default_factory=list)


class Evidence(GovernanceFields):
    id: str
    source_id: str
    quote: str
    page_number: int | None = None
    url: str | None = None
    section: str | None = None
    source_mode: Literal["native_text", "ocr_text", "timeout_placeholder", "failed_placeholder", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)


class Claim(GovernanceFields):
    id: str
    text: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["active", "uncertain", "contradicted", "outdated"] = "active"
    created_at: datetime = Field(default_factory=utc_now)


class Entity(GovernanceFields):
    id: str
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class Relation(GovernanceFields):
    id: str
    subject_id: str
    predicate: str
    object_id: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class EmbeddingRecord(BaseModel):
    id: str
    record_type: Literal["text_unit", "wiki_page", "claim", "evidence", "entity", "relation"]
    record_id: str
    embedding: list[float]
    model: str
    dimensions: int
    text: str
    created_at: datetime = Field(default_factory=utc_now)


class AuditEvent(BaseModel):
    id: str
    event_type: Literal["create", "update", "delete", "proposal", "export", "apply"]
    target_type: str
    target_id: str
    actor: str = "system"
    source: str = "llm-kg"
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)


class VerificationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class VerificationResult(BaseModel):
    target_type: str
    target_id: str
    valid: bool
    review_state: str | None = None
    confidence: float | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    issues: list[VerificationIssue] = Field(default_factory=list)


class TraceNode(BaseModel):
    kind: str
    id: str
    title: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceResult(BaseModel):
    target_type: str
    target_id: str
    nodes: list[TraceNode] = Field(default_factory=list)
    gaps: list[VerificationIssue] = Field(default_factory=list)


class UpdateProposalDraft(BaseModel):
    id: str
    proposal_type: str
    target_type: str
    target_id: str | None = None
    title: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_signal_ids: list[str] = Field(default_factory=list)
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    status: str = "draft"
    created_at: datetime = Field(default_factory=utc_now)


class OntologySchema(BaseModel):
    id: str = "generic"
    entity_types: list[str] = Field(default_factory=list)
    relation_predicates: list[str] = Field(default_factory=list)
    require_claim_evidence: bool = True
    require_relation_trace: bool = True
    updated_at: datetime = Field(default_factory=utc_now)


class ApplyResult(BaseModel):
    status: Literal["applied", "rejected"]
    message: str
    audit_event_ids: list[str] = Field(default_factory=list)


class LintIssue(BaseModel):
    code: str
    message: str
    path: str | None = None
    severity: Literal["info", "warning", "error"] = "warning"


class QueryHit(BaseModel):
    kind: Literal["wiki", "claim", "evidence", "text_unit", "entity", "relation"]
    id: str
    title: str | None = None
    text: str
    score: float
    path: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    id: str
    order: int
    description: str
    claim_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ReasoningTrace(BaseModel):
    id: str
    question: str
    answer: str
    mode: Literal["basic", "local"] = "local"
    hits: list[QueryHit] = Field(default_factory=list)
    used_claim_ids: list[str] = Field(default_factory=list)
    used_relation_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    decision_output: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class QueryResult(BaseModel):
    question: str
    mode: Literal["basic", "local"] = "local"
    answer: str
    hits: list[QueryHit]
    evidence: list[Evidence] = Field(default_factory=list)
    trace_id: str | None = None


class IngestResult(BaseModel):
    document: Document
    text_units: list[TextUnit] = Field(default_factory=list)
    wiki_page: WikiPage
    wiki_pages: list[WikiPage] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    proposals: list[UpdateProposalDraft] = Field(default_factory=list)


def relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)
