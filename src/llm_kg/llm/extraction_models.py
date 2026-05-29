from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedEvidence(BaseModel):
    id: str = Field(description="Stable evidence id, preferably ev_<hash>.")
    source_id: str
    quote: str
    page_number: int | None = None
    section: str | None = None
    source_mode: Literal["native_text", "ocr_text", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedClaim(BaseModel):
    id: str = Field(description="Stable claim id, preferably claim_<hash>.")
    text: str
    source_ids: list[str]
    evidence_ids: list[str]
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ExtractedEntity(BaseModel):
    id: str = Field(description="Stable entity id, preferably entity_<hash>.")
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    source_ids: list[str]


class ExtractedRelation(BaseModel):
    id: str = Field(description="Stable relation id, preferably rel_<hash>.")
    subject_id: str
    predicate: str
    object_id: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class KnowledgeExtractionOutput(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)
    evidence: list[ExtractedEvidence] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
