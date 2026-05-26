from __future__ import annotations

from collections import Counter
from pathlib import Path

from llm_kg.models import Claim, Entity, Evidence, LintIssue, Relation
from llm_kg.storage import JsonlStore, MarkdownStore


def lint_workspace(workspace: Path) -> list[LintIssue]:
    workspace = workspace.resolve()
    issues: list[LintIssue] = []
    markdown = MarkdownStore(workspace)
    pages = markdown.load_pages()
    page_titles = {page.title for page in pages}
    page_stems = {Path(page.path).stem for page in pages}

    for page in pages:
        if not page.content_md.strip():
            issues.append(LintIssue(code="empty_wiki_page", message="Wiki page is empty.", path=page.path))
        for link in page.wikilinks:
            if link not in page_titles and link not in page_stems:
                issues.append(
                    LintIssue(
                        code="broken_wikilink",
                        message=f"Wikilink does not resolve: [[{link}]]",
                        path=page.path,
                    )
                )

    jsonl = JsonlStore(workspace)
    claims = jsonl.load("claims.jsonl", Claim)
    evidence = jsonl.load("evidence.jsonl", Evidence)
    entities = jsonl.load("nodes.jsonl", Entity)
    relations = jsonl.load("edges.jsonl", Relation)

    evidence_ids = {item.id for item in evidence}
    entity_ids = {item.id for item in entities}
    claim_ids = {item.id for item in claims}

    for claim in claims:
        if not claim.evidence_ids:
            issues.append(LintIssue(code="claim_missing_evidence", message=f"Claim has no evidence: {claim.id}"))
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_ids:
                issues.append(
                    LintIssue(code="claim_bad_evidence_ref", message=f"Claim {claim.id} references {evidence_id}")
                )

    name_counts = Counter(entity.name.lower() for entity in entities)
    for name, count in name_counts.items():
        if count > 1:
            issues.append(LintIssue(code="duplicate_entity_name", message=f"Duplicate entity name: {name}"))

    for relation in relations:
        if relation.subject_id not in entity_ids:
            issues.append(
                LintIssue(
                    code="relation_bad_subject_ref",
                    message=f"Relation {relation.id} references missing subject {relation.subject_id}",
                )
            )
        if relation.object_id not in entity_ids:
            issues.append(
                LintIssue(
                    code="relation_bad_object_ref",
                    message=f"Relation {relation.id} references missing object {relation.object_id}",
                )
            )
        for claim_id in relation.claim_ids:
            if claim_id not in claim_ids:
                issues.append(
                    LintIssue(code="relation_bad_claim_ref", message=f"Relation {relation.id} references {claim_id}")
                )
        for evidence_id in relation.evidence_ids:
            if evidence_id not in evidence_ids:
                issues.append(
                    LintIssue(
                        code="relation_bad_evidence_ref",
                        message=f"Relation {relation.id} references {evidence_id}",
                    )
                )
    return issues
