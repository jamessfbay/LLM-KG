from pathlib import Path

from llm_kg.governance import apply_update_plan, create_proposal, export_proposal, trace_object, verify_claim
from llm_kg.models import Claim, Entity, Evidence, Relation
from llm_kg.storage import JsonlStore


def seed_governance_workspace(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert("evidence.jsonl", [Evidence(id="ev_1", source_id="doc_1", quote="A requires C.", confidence=0.9)])
    store.upsert(
        "claims.jsonl",
        [
            Claim(
                id="claim_1",
                text="A requires C.",
                source_ids=["doc_1"],
                evidence_ids=["ev_1"],
                subject="A",
                predicate="requires",
                object="C",
                confidence=0.9,
            )
        ],
    )
    store.upsert(
        "nodes.jsonl",
        [
            Entity(id="entity_a", name="A", entity_type="concept", source_ids=["doc_1"]),
            Entity(id="entity_c", name="C", entity_type="concept", source_ids=["doc_1"]),
        ],
    )
    store.upsert(
        "edges.jsonl",
        [
            Relation(
                id="rel_1",
                subject_id="entity_a",
                predicate="requires",
                object_id="entity_c",
                claim_ids=["claim_1"],
                evidence_ids=["ev_1"],
                confidence=0.9,
            )
        ],
    )


def test_governance_defaults_serialize() -> None:
    claim = Claim(id="claim_1", text="A requires C.", confidence=0.9)

    payload = claim.model_dump(mode="json")

    assert payload["review_state"] == "auto_accepted"
    assert payload["version"] == 1
    assert payload["created_by"] == "system"


def test_verify_claim_with_evidence_passes(tmp_path: Path) -> None:
    seed_governance_workspace(tmp_path)

    result = verify_claim("claim_1", tmp_path)

    assert result.valid is True
    assert result.evidence[0].id == "ev_1"
    assert result.issues == []


def test_verify_claim_missing_evidence_reports_issue(tmp_path: Path) -> None:
    JsonlStore(tmp_path).upsert(
        "claims.jsonl",
        [Claim(id="claim_1", text="A requires C.", evidence_ids=["missing"], confidence=0.6)],
    )

    result = verify_claim("claim_1", tmp_path)

    assert result.valid is False
    assert {issue.code for issue in result.issues} == {"claim_bad_evidence_ref"}


def test_trace_claim_returns_source_evidence_and_relation(tmp_path: Path) -> None:
    seed_governance_workspace(tmp_path)

    result = trace_object("claim", "claim_1", tmp_path)

    assert {node.kind for node in result.nodes} >= {"claim", "source", "evidence", "relation"}
    assert result.gaps == []


def test_create_and_export_proposal_matches_llm_kee_shape(tmp_path: Path) -> None:
    seed_governance_workspace(tmp_path)

    proposal = create_proposal(
        "claim",
        "claim_1",
        {"text": "A requires C.", "evidence_ids": ["ev_1"], "rationale": "Corrected from evidence."},
        tmp_path,
    )
    exported = export_proposal(proposal.id, tmp_path)

    assert set(exported) >= {
        "id",
        "proposal_type",
        "target_type",
        "target_id",
        "title",
        "rationale",
        "evidence_ids",
        "source_signal_ids",
        "proposed_change",
        "confidence",
        "status",
    }
    assert exported["target_type"] == "claim"
    assert exported["evidence_ids"] == ["ev_1"]


def test_apply_update_plan_requires_approval_and_updates_claim(tmp_path: Path) -> None:
    seed_governance_workspace(tmp_path)

    rejected = apply_update_plan(
        {"target_type": "claim", "target_id": "claim_1", "payload": {"change": {"text": "A requires D."}}},
        tmp_path,
    )
    assert rejected.status == "rejected"

    applied = apply_update_plan(
        {
            "status": "approved",
            "target_type": "claim",
            "target_id": "claim_1",
            "payload": {"change": {"text": "A requires D.", "evidence_ids": ["ev_1"]}},
        },
        tmp_path,
    )

    claim = JsonlStore(tmp_path).load("claims.jsonl", Claim)[0]
    assert applied.status == "applied"
    assert claim.text == "A requires D."
    assert claim.review_state == "approved"
    assert claim.version == 2
