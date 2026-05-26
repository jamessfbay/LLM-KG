from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


class WikiPage(BaseModel):
    id: str
    title: str
    page_type: Literal["source", "entity", "concept", "synthesis", "comparison"]
    path: str
    content_md: str
    source_ids: list[str] = Field(default_factory=list)
    wikilinks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class Evidence(BaseModel):
    id: str
    source_id: str
    quote: str
    page_number: int | None = None
    url: str | None = None
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class Claim(BaseModel):
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


class Entity(BaseModel):
    id: str
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    id: str
    subject_id: str
    predicate: str
    object_id: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class LintIssue(BaseModel):
    code: str
    message: str
    path: str | None = None
    severity: Literal["info", "warning", "error"] = "warning"


class QueryHit(BaseModel):
    kind: Literal["wiki", "claim", "evidence"]
    id: str
    title: str | None = None
    text: str
    score: float
    path: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    question: str
    answer: str
    hits: list[QueryHit]
    evidence: list[Evidence] = Field(default_factory=list)


class IngestResult(BaseModel):
    document: Document
    wiki_page: WikiPage
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


def relative_to_workspace(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)
