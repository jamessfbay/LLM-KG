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
    WikiPage,
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


def verify_object(target_type: str, target_id: str, workspace: Path) -> VerificationResult:
    if target_type == "claim":
        return verify_claim(target_id, workspace)
    settings = Settings.from_env(workspace)
    issues: list[VerificationIssue] = []
    evidence: list[Evidence] = []

    if target_type == "relation":
        relation = _get_relation(target_id, workspace, settings)
        if not relation:
            return _not_found_result(target_type, target_id)
        if not _get_entity(relation.subject_id, workspace, settings):
            issues.append(VerificationIssue(code="relation_missing_subject", message=f"Missing subject entity: {relation.subject_id}", severity="error"))
        if not _get_entity(relation.object_id, workspace, settings):
            issues.append(VerificationIssue(code="relation_missing_object", message=f"Missing object entity: {relation.object_id}", severity="error"))
        if not relation.claim_ids and not relation.evidence_ids:
            issues.append(VerificationIssue(code="relation_missing_trace", message="Relation has no claim or evidence trace.", severity="error"))
        for claim_id in relation.claim_ids:
            if not _get_claim(claim_id, workspace, settings):
                issues.append(VerificationIssue(code="relation_bad_claim_ref", message=f"Missing claim: {claim_id}", severity="error"))
        evidence = _get_evidence_many(relation.evidence_ids, workspace, settings)
        _append_missing_evidence_issues(issues, relation.evidence_ids, evidence, "relation_bad_evidence_ref")
        return VerificationResult(
            target_type=target_type,
            target_id=target_id,
            valid=not any(issue.severity == "error" for issue in issues),
            review_state=relation.review_state,
            confidence=relation.confidence,
            evidence=evidence,
            issues=issues,
        )

    if target_type == "entity":
        entity = _get_entity(target_id, workspace, settings)
        if not entity:
            return _not_found_result(target_type, target_id)
        if not entity.source_ids:
            issues.append(VerificationIssue(code="entity_missing_source", message="Entity has no source IDs.", severity="warning"))
        return VerificationResult(target_type=target_type, target_id=target_id, valid=True, review_state=entity.review_state, issues=issues)

    if target_type == "evidence":
        items = _get_evidence_many([target_id], workspace, settings)
        if not items:
            return _not_found_result(target_type, target_id)
        item = items[0]
        if not item.source_id:
            issues.append(VerificationIssue(code="evidence_missing_source", message="Evidence has no source ID.", severity="error"))
        if not item.quote.strip():
            issues.append(VerificationIssue(code="evidence_missing_quote", message="Evidence quote is empty.", severity="error"))
        return VerificationResult(
            target_type=target_type,
            target_id=target_id,
            valid=not any(issue.severity == "error" for issue in issues),
            review_state=item.review_state,
            confidence=item.confidence,
            evidence=[item],
            issues=issues,
        )

    if target_type == "wiki_page":
        page = _get_wiki_page(target_id, workspace)
        if not page:
            return _not_found_result(target_type, target_id)
        if not page.content_md.strip():
            issues.append(VerificationIssue(code="wiki_page_empty", message="Wiki page content is empty.", severity="error"))
        return VerificationResult(
            target_type=target_type,
            target_id=target_id,
            valid=not any(issue.severity == "error" for issue in issues),
            review_state=page.review_state,
            issues=issues,
        )

    raise ValueError("target_type must be one of: claim, relation, entity, evidence, wiki_page")


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
    proposal_type: str | None = None,
) -> UpdateProposalDraft:
    settings = Settings.from_env(workspace)
    evidence_ids = list(proposed_change.get("evidence_ids") or [])
    if not evidence_ids and target_type == "claim":
        claim = _get_claim(target_id, workspace, settings)
        evidence_ids = claim.evidence_ids if claim else []
    proposal_kind = proposal_type or _default_proposal_type(target_type)
    proposal = UpdateProposalDraft(
        id=stable_id("prop", json.dumps([target_type, target_id, proposed_change], sort_keys=True, default=str)),
        proposal_type=proposal_kind,
        target_type=target_type,
        target_id=target_id,
        title=proposed_change.get("title") or f"Proposed {proposal_kind} for {target_type} {target_id}",
        rationale=proposed_change.get("rationale") or "Generated by LLM-KG evidence governance workflow.",
        evidence_ids=evidence_ids,
        proposed_change=proposed_change,
        confidence=float(proposed_change.get("confidence", 0.5)),
        status=str(proposed_change.get("proposal_status") or "draft"),
    )
    JsonlStore(workspace).upsert("proposals.jsonl", [proposal])
    _record_audit(workspace, "proposal", target_type, target_id, None, proposal.model_dump(mode="json"))
    postgres = build_postgres_store(settings)
    if postgres:
        postgres.upsert_proposal(proposal)
    return proposal


def _default_proposal_type(target_type: str) -> str:
    return {
        "claim": "update_claim",
        "relation": "update_relation",
        "entity": "merge_entity",
        "evidence": "add_evidence",
        "wiki_page": "update_wiki_page",
    }.get(target_type, "update_claim")


def export_proposal(proposal_id: str, workspace: Path, export_format: str = "llm-kee") -> dict[str, Any]:
    if export_format != "llm-kee":
        raise ValueError("Only llm-kee export format is supported.")
    proposal = _get_proposal(proposal_id, workspace)
    if not proposal:
        raise ValueError(f"Proposal not found: {proposal_id}")
    payload = proposal.model_dump(mode="json")
    _record_audit(workspace, "export", proposal.target_type, proposal.target_id or proposal.id, None, payload)
    return payload


def import_kee_decision(decision: dict[str, Any], workspace: Path) -> dict[str, Any]:
    proposal_id = str(decision.get("proposal_id") or decision.get("id") or "unknown")
    event = _record_audit(workspace, "update", "kee_decision", proposal_id, None, decision)
    return {"status": "imported", "audit_event_id": event.id, "proposal_id": proposal_id}


def apply_kee_plan(plan: dict[str, Any], workspace: Path) -> ApplyResult:
    return apply_update_plan(plan, workspace)


def apply_update_plan(plan: dict[str, Any], workspace: Path) -> ApplyResult:
    if not _is_approved_plan(plan):
        return ApplyResult(status="rejected", message="Update plan is not approved by LLM-KEE.")

    target_type = str(plan.get("target_type") or "")
    target_id = plan.get("target_id")
    payload = dict(plan.get("payload") or {})
    change = dict(payload.get("change") or payload.get("proposed_change") or {})
    if "new_value" in change and isinstance(change["new_value"], dict):
        change = dict(change["new_value"])
    if target_type not in {"claim", "relation", "entity", "evidence", "wiki_page"} or not target_id:
        return ApplyResult(status="rejected", message="Only approved claim/relation/entity/evidence/wiki_page update plans are supported.")

    settings = Settings.from_env(workspace)
    existing = _get_governed_object(target_type, str(target_id), workspace, settings)
    if not existing:
        return ApplyResult(status="rejected", message=f"{target_type} not found: {target_id}")

    before = existing.model_dump(mode="json")
    operation = str(plan.get("operation") or plan.get("proposal_type") or "").lower()
    allowed_fields = _allowed_apply_fields(target_type)
    for field in allowed_fields:
        if field in change:
            setattr(existing, field, change[field])
    if operation.startswith("retire_") or change.get("review_state") == "superseded":
        existing.review_state = "superseded"
        if hasattr(existing, "status"):
            setattr(existing, "status", "outdated")
    else:
        existing.review_state = "approved"
    if "supersedes_id" in change:
        existing.supersedes_id = change["supersedes_id"]
    if "superseded_by_id" in change:
        existing.superseded_by_id = change["superseded_by_id"]
    existing.version += 1
    existing.updated_by = "llm-kee"
    existing.updated_at = utc_now()
    _save_governed_object(target_type, existing, workspace)
    postgres_event_id = None
    postgres = build_postgres_store(settings)
    if postgres:
        if target_type == "claim":
            postgres_event_id = postgres.update_claim(existing, before=before)
        elif target_type == "relation":
            postgres_event_id = postgres.update_relation(existing, before=before)
        elif target_type == "entity":
            postgres_event_id = postgres.update_entity(existing, before=before)
        elif target_type == "evidence":
            postgres_event_id = postgres.update_evidence(existing, before=before)
        elif target_type == "wiki_page":
            postgres_event_id = postgres.update_wiki_page(existing, before=before)
    event = _record_audit(workspace, "apply", target_type, existing.id, before, existing.model_dump(mode="json"))
    event_ids = [event.id]
    if postgres_event_id:
        event_ids.append(postgres_event_id)
    return ApplyResult(status="applied", message=f"Applied approved update plan to {target_type} {existing.id}.", audit_event_ids=event_ids)


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


def _get_wiki_page(page_id: str, workspace: Path) -> WikiPage | None:
    postgres = build_postgres_store(Settings.from_env(workspace))
    if postgres:
        page = postgres.get_wiki_page(page_id)
        if page:
            return page
    return next((page for page in JsonlStore(workspace).load("wiki_pages.jsonl", WikiPage) if page.id == page_id), None)


def _get_governed_object(target_type: str, target_id: str, workspace: Path, settings: Settings):
    if target_type == "claim":
        return _get_claim(target_id, workspace, settings)
    if target_type == "relation":
        return _get_relation(target_id, workspace, settings)
    if target_type == "entity":
        return _get_entity(target_id, workspace, settings)
    if target_type == "evidence":
        items = _get_evidence_many([target_id], workspace, settings)
        return items[0] if items else None
    if target_type == "wiki_page":
        return _get_wiki_page(target_id, workspace)
    return None


def _save_governed_object(target_type: str, obj, workspace: Path) -> None:
    filename = {
        "claim": "claims.jsonl",
        "relation": "edges.jsonl",
        "entity": "nodes.jsonl",
        "evidence": "evidence.jsonl",
        "wiki_page": "wiki_pages.jsonl",
    }[target_type]
    JsonlStore(workspace).upsert(filename, [obj])
    if target_type == "wiki_page":
        path = workspace / obj.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obj.content_md, encoding="utf-8")


def _allowed_apply_fields(target_type: str) -> tuple[str, ...]:
    return {
        "claim": ("text", "source_ids", "evidence_ids", "subject", "predicate", "object", "confidence", "status", "governance_notes"),
        "relation": ("subject_id", "predicate", "object_id", "claim_ids", "evidence_ids", "confidence", "valid_from", "valid_to", "governance_notes"),
        "entity": ("name", "entity_type", "aliases", "description", "source_ids", "governance_notes"),
        "evidence": ("source_id", "quote", "page_number", "url", "section", "confidence", "governance_notes"),
        "wiki_page": ("title", "page_type", "path", "content_md", "source_ids", "wikilinks", "tags", "governance_notes"),
    }[target_type]


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


def _not_found_result(target_type: str, target_id: str) -> VerificationResult:
    return VerificationResult(
        target_type=target_type,
        target_id=target_id,
        valid=False,
        issues=[VerificationIssue(code="object_not_found", message=f"{target_type} not found: {target_id}", severity="error")],
    )


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


def _append_missing_evidence_issues(
    issues: list[VerificationIssue],
    expected_ids: list[str],
    evidence: list[Evidence],
    code: str,
) -> None:
    found = {item.id for item in evidence}
    for evidence_id in sorted(set(expected_ids) - found):
        issues.append(
            VerificationIssue(
                code=code,
                message=f"Missing evidence: {evidence_id}",
                severity="error",
            )
        )


def _is_approved_plan(plan: dict[str, Any]) -> bool:
    approved_values = {"approved", "auto_apply", "applied"}
    return bool(plan.get("approved")) or str(plan.get("status", "")).lower() in approved_values or str(
        plan.get("decision", "")
    ).lower() in approved_values or bool((plan.get("payload") or {}).get("approved"))
