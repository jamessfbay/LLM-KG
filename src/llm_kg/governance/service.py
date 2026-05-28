from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_kg.config import Settings
from llm_kg.models import (
    ApplyResult,
    AuditEvent,
    Claim,
    Entity,
    Evidence,
    Relation,
    TraceNode,
    TraceResult,
    UpdateProposalDraft,
    VerificationIssue,
    VerificationResult,
)
from llm_kg.models.core import utc_now
from llm_kg.readers.common import stable_id
from llm_kg.storage import JsonlStore, build_postgres_store


def verify_claim(claim_id: str, workspace: Path) -> VerificationResult:
    settings = Settings.from_env(workspace)
    claim = _get_claim(claim_id, workspace, settings)
    if not claim:
        return VerificationResult(
            target_type="claim",
            target_id=claim_id,
            valid=False,
            issues=[VerificationIssue(code="claim_not_found", message=f"Claim not found: {claim_id}", severity="error")],
        )

    evidence = _get_evidence_many(claim.evidence_ids, workspace, settings)
    issues: list[VerificationIssue] = []
    if not claim.evidence_ids:
        issues.append(VerificationIssue(code="claim_missing_evidence", message="Claim has no evidence IDs.", severity="error"))
    missing = sorted(set(claim.evidence_ids) - {item.id for item in evidence})
    for evidence_id in missing:
        issues.append(
            VerificationIssue(
                code="claim_bad_evidence_ref",
                message=f"Claim references missing evidence: {evidence_id}",
                severity="error",
            )
        )
    if claim.review_state in {"rejected", "superseded"}:
        issues.append(
            VerificationIssue(
                code=f"claim_{claim.review_state}",
                message=f"Claim review_state is {claim.review_state}.",
                severity="warning",
            )
        )
    return VerificationResult(
        target_type="claim",
        target_id=claim.id,
        valid=not any(issue.severity == "error" for issue in issues),
        review_state=claim.review_state,
        confidence=claim.confidence,
        evidence=evidence,
        issues=issues,
    )


def trace_object(target_type: str, target_id: str, workspace: Path) -> TraceResult:
    settings = Settings.from_env(workspace)
    nodes: list[TraceNode] = []
    gaps: list[VerificationIssue] = []

    if target_type == "claim":
        claim = _get_claim(target_id, workspace, settings)
        if not claim:
            return _missing_trace(target_type, target_id)
        nodes.append(TraceNode(kind="claim", id=claim.id, text=claim.text, metadata=claim.model_dump(mode="json")))
        for source_id in claim.source_ids:
            nodes.append(TraceNode(kind="source", id=source_id))
        evidence = _get_evidence_many(claim.evidence_ids, workspace, settings)
        _append_evidence_nodes(nodes, evidence)
        _missing_evidence_gaps(gaps, claim.evidence_ids, evidence)
        for relation in _get_relations_for_claim(claim.id, workspace, settings):
            nodes.append(
                TraceNode(
                    kind="relation",
                    id=relation.id,
                    text=f"{relation.subject_id} -> {relation.predicate} -> {relation.object_id}",
                    metadata=relation.model_dump(mode="json"),
                )
            )
        return TraceResult(target_type=target_type, target_id=target_id, nodes=nodes, gaps=gaps)

    if target_type == "relation":
        relation = _get_relation(target_id, workspace, settings)
        if not relation:
            return _missing_trace(target_type, target_id)
        nodes.append(
            TraceNode(
                kind="relation",
                id=relation.id,
                text=f"{relation.subject_id} -> {relation.predicate} -> {relation.object_id}",
                metadata=relation.model_dump(mode="json"),
            )
        )
        for entity_id in [relation.subject_id, relation.object_id]:
            entity = _get_entity(entity_id, workspace, settings)
            if entity:
                nodes.append(TraceNode(kind="entity", id=entity.id, title=entity.name, metadata=entity.model_dump(mode="json")))
            else:
                gaps.append(VerificationIssue(code="relation_missing_entity", message=f"Missing entity: {entity_id}", severity="error"))
        for claim_id in relation.claim_ids:
            claim = _get_claim(claim_id, workspace, settings)
            if claim:
                nodes.append(TraceNode(kind="claim", id=claim.id, text=claim.text, metadata=claim.model_dump(mode="json")))
        evidence = _get_evidence_many(relation.evidence_ids, workspace, settings)
        _append_evidence_nodes(nodes, evidence)
        _missing_evidence_gaps(gaps, relation.evidence_ids, evidence)
        return TraceResult(target_type=target_type, target_id=target_id, nodes=nodes, gaps=gaps)

    if target_type == "entity":
        entity = _get_entity(target_id, workspace, settings)
        if not entity:
            return _missing_trace(target_type, target_id)
        nodes.append(TraceNode(kind="entity", id=entity.id, title=entity.name, metadata=entity.model_dump(mode="json")))
        return TraceResult(target_type=target_type, target_id=target_id, nodes=nodes, gaps=gaps)

    if target_type == "evidence":
        evidence = _get_evidence_many([target_id], workspace, settings)
        if not evidence:
            return _missing_trace(target_type, target_id)
        _append_evidence_nodes(nodes, evidence)
        nodes.append(TraceNode(kind="source", id=evidence[0].source_id))
        return TraceResult(target_type=target_type, target_id=target_id, nodes=nodes, gaps=gaps)

    raise ValueError("target_type must be one of: claim, relation, entity, evidence")


def create_proposal(
    target_type: str,
    target_id: str,
    proposed_change: dict[str, Any],
    workspace: Path,
    proposal_type: str = "correction",
) -> UpdateProposalDraft:
    settings = Settings.from_env(workspace)
    evidence_ids = list(proposed_change.get("evidence_ids") or [])
    if not evidence_ids and target_type == "claim":
        claim = _get_claim(target_id, workspace, settings)
        evidence_ids = claim.evidence_ids if claim else []
    proposal = UpdateProposalDraft(
        id=stable_id("prop", json.dumps([target_type, target_id, proposed_change], sort_keys=True, default=str)),
        proposal_type=proposal_type,
        target_type=target_type,
        target_id=target_id,
        title=proposed_change.get("title") or f"Proposed {proposal_type} for {target_type} {target_id}",
        rationale=proposed_change.get("rationale") or "Generated by LLM-KG evidence governance workflow.",
        evidence_ids=evidence_ids,
        proposed_change=proposed_change,
        confidence=float(proposed_change.get("confidence", 0.5)),
        status="draft",
    )
    JsonlStore(workspace).upsert("proposals.jsonl", [proposal])
    _record_audit(workspace, "proposal", target_type, target_id, None, proposal.model_dump(mode="json"))
    postgres = build_postgres_store(settings)
    if postgres:
        postgres.upsert_proposal(proposal)
    return proposal


def export_proposal(proposal_id: str, workspace: Path, export_format: str = "llm-kee") -> dict[str, Any]:
    if export_format != "llm-kee":
        raise ValueError("Only llm-kee export format is supported.")
    proposal = _get_proposal(proposal_id, workspace)
    if not proposal:
        raise ValueError(f"Proposal not found: {proposal_id}")
    payload = proposal.model_dump(mode="json")
    _record_audit(workspace, "export", proposal.target_type, proposal.target_id or proposal.id, None, payload)
    return payload


def apply_update_plan(plan: dict[str, Any], workspace: Path) -> ApplyResult:
    if not _is_approved_plan(plan):
        return ApplyResult(status="rejected", message="Update plan is not approved by LLM-KEE.")

    target_type = str(plan.get("target_type") or "")
    target_id = plan.get("target_id")
    payload = dict(plan.get("payload") or {})
    change = dict(payload.get("change") or payload.get("proposed_change") or {})
    if target_type != "claim" or not target_id:
        return ApplyResult(status="rejected", message="Only approved claim update plans are supported in this phase.")

    settings = Settings.from_env(workspace)
    claims = JsonlStore(workspace).load("claims.jsonl", Claim)
    existing = next((claim for claim in claims if claim.id == target_id), None) or _get_claim(target_id, workspace, settings)
    if not existing:
        return ApplyResult(status="rejected", message=f"Claim not found: {target_id}")

    before = existing.model_dump(mode="json")
    for field in ("text", "evidence_ids", "subject", "predicate", "object", "confidence", "status", "governance_notes"):
        if field in change:
            setattr(existing, field, change[field])
    existing.version += 1
    existing.review_state = "approved"
    existing.updated_by = "llm-kee"
    existing.updated_at = utc_now()
    if any(claim.id == existing.id for claim in claims):
        JsonlStore(workspace).upsert("claims.jsonl", [existing])
    postgres_event_id = None
    postgres = build_postgres_store(settings)
    if postgres:
        postgres_event_id = postgres.update_claim(existing, before=before)
    event = _record_audit(workspace, "apply", "claim", existing.id, before, existing.model_dump(mode="json"))
    event_ids = [event.id]
    if postgres_event_id:
        event_ids.append(postgres_event_id)
    return ApplyResult(status="applied", message=f"Applied approved update plan to claim {existing.id}.", audit_event_ids=event_ids)


def _get_claim(claim_id: str, workspace: Path, settings: Settings) -> Claim | None:
    postgres = build_postgres_store(settings)
    if postgres:
        claim = postgres.get_claim(claim_id)
        if claim:
            return claim
    return next((claim for claim in JsonlStore(workspace).load("claims.jsonl", Claim) if claim.id == claim_id), None)


def _get_evidence_many(evidence_ids: list[str], workspace: Path, settings: Settings) -> list[Evidence]:
    postgres = build_postgres_store(settings)
    if postgres:
        evidence = postgres.get_evidence_many(evidence_ids)
        if evidence:
            return evidence
    all_evidence = JsonlStore(workspace).load("evidence.jsonl", Evidence)
    wanted = set(evidence_ids)
    return [item for item in all_evidence if item.id in wanted]


def _get_relations_for_claim(claim_id: str, workspace: Path, settings: Settings) -> list[Relation]:
    postgres = build_postgres_store(settings)
    if postgres:
        relations = postgres.get_relations_for_claim(claim_id)
        if relations:
            return relations
    return [relation for relation in JsonlStore(workspace).load("edges.jsonl", Relation) if claim_id in relation.claim_ids]


def _get_relation(relation_id: str, workspace: Path, settings: Settings) -> Relation | None:
    postgres = build_postgres_store(settings)
    if postgres:
        relation = postgres.get_relation(relation_id)
        if relation:
            return relation
    return next((relation for relation in JsonlStore(workspace).load("edges.jsonl", Relation) if relation.id == relation_id), None)


def _get_entity(entity_id: str, workspace: Path, settings: Settings) -> Entity | None:
    postgres = build_postgres_store(settings)
    if postgres:
        entity = postgres.get_entity(entity_id)
        if entity:
            return entity
    return next((entity for entity in JsonlStore(workspace).load("nodes.jsonl", Entity) if entity.id == entity_id), None)


def _get_proposal(proposal_id: str, workspace: Path) -> UpdateProposalDraft | None:
    settings = Settings.from_env(workspace)
    postgres = build_postgres_store(settings)
    if postgres:
        proposal = postgres.get_proposal(proposal_id)
        if proposal:
            return proposal
    return next(
        (proposal for proposal in JsonlStore(workspace).load("proposals.jsonl", UpdateProposalDraft) if proposal.id == proposal_id),
        None,
    )


def _record_audit(
    workspace: Path,
    event_type: str,
    target_type: str,
    target_id: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> AuditEvent:
    event = AuditEvent(
        id=stable_id(
            "audit",
            json.dumps([event_type, target_type, target_id, before, after], sort_keys=True, default=str),
        ),
        event_type=event_type,  # type: ignore[arg-type]
        target_type=target_type,
        target_id=target_id,
        before=before,
        after=after,
    )
    JsonlStore(workspace).upsert("audit_events.jsonl", [event])
    postgres = build_postgres_store(Settings.from_env(workspace))
    if postgres:
        postgres.insert_audit_event(event)
    return event


def _missing_trace(target_type: str, target_id: str) -> TraceResult:
    return TraceResult(
        target_type=target_type,
        target_id=target_id,
        gaps=[VerificationIssue(code="object_not_found", message=f"{target_type} not found: {target_id}", severity="error")],
    )


def _append_evidence_nodes(nodes: list[TraceNode], evidence: list[Evidence]) -> None:
    for item in evidence:
        nodes.append(TraceNode(kind="evidence", id=item.id, text=item.quote, metadata=item.model_dump(mode="json")))


def _missing_evidence_gaps(gaps: list[VerificationIssue], expected_ids: list[str], evidence: list[Evidence]) -> None:
    found = {item.id for item in evidence}
    for evidence_id in sorted(set(expected_ids) - found):
        gaps.append(
            VerificationIssue(
                code="missing_evidence",
                message=f"Evidence not found: {evidence_id}",
                severity="error",
            )
        )


def _is_approved_plan(plan: dict[str, Any]) -> bool:
    approved_values = {"approved", "auto_apply", "applied"}
    return bool(plan.get("approved")) or str(plan.get("status", "")).lower() in approved_values or str(
        plan.get("decision", "")
    ).lower() in approved_values or bool((plan.get("payload") or {}).get("approved"))
