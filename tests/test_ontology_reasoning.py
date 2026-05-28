from pathlib import Path

from llm_kg.api import export_reasoning_trace, list_reasoning_traces, query_knowledge, verify_object
from llm_kg.governance import apply_update_plan
from llm_kg.llm.mock_client import MockLLMClient
from llm_kg.models import Claim, Entity, Evidence, Relation, UpdateProposalDraft, WikiPage
from llm_kg.ontology import build_ontology_registry
from llm_kg.pipeline.ingest import ingest_source
from llm_kg.storage import JsonlStore


def test_ontology_registry_flags_unknown_values() -> None:
    registry = build_ontology_registry("generic")
    entity = Entity(id="entity_1", name="Thing", entity_type="surprise")
    relation = Relation(id="rel_1", subject_id="entity_1", predicate="mystery", object_id="missing", confidence=0.4)

    assert registry.validate_entity(entity)[0].code == "unknown_entity_type"
    codes = {issue.code for issue in registry.validate_relation(relation, {"entity_1"})}
    assert codes == {"unknown_relation_predicate", "relation_missing_object"}


def test_ingest_governance_generates_compiled_wiki_and_filters_bad_records(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text(
        "Project Alpha requires Policy Beta. Policy Beta affects Project Alpha.",
        encoding="utf-8",
    )

    result = ingest_source(source, MockLLMClient(), tmp_path)

    page_types = {page.page_type for page in result.wiki_pages}
    assert {"source", "entity", "concept", "synthesis"}.issubset(page_types)
    assert result.claims
    assert all(claim.evidence_ids for claim in result.claims)
    assert JsonlStore(tmp_path).load("wiki_pages.jsonl", WikiPage)


def test_query_persists_and_exports_reasoning_trace(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert("evidence.jsonl", [Evidence(id="ev_1", source_id="doc_1", quote="Alpha requires Beta.", confidence=0.9)])
    store.upsert(
        "claims.jsonl",
        [Claim(id="claim_1", text="Alpha requires Beta.", evidence_ids=["ev_1"], confidence=0.9)],
    )

    result = query_knowledge("What does Alpha require?", workspace=tmp_path, persist_trace=True)

    assert result.trace_id
    traces = list_reasoning_traces(tmp_path)
    assert traces[0].id == result.trace_id
    exported = export_reasoning_trace(result.trace_id, tmp_path)
    assert exported["question"] == "What does Alpha require?"
    assert "final_answer" in exported


def test_verify_relation_and_apply_relation_update(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    store.upsert("evidence.jsonl", [Evidence(id="ev_1", source_id="doc_1", quote="A supports B.", confidence=0.9)])
    store.upsert("claims.jsonl", [Claim(id="claim_1", text="A supports B.", evidence_ids=["ev_1"], confidence=0.9)])
    store.upsert(
        "nodes.jsonl",
        [
            Entity(id="entity_a", name="A", entity_type="concept"),
            Entity(id="entity_b", name="B", entity_type="concept"),
        ],
    )
    store.upsert(
        "edges.jsonl",
        [
            Relation(
                id="rel_1",
                subject_id="entity_a",
                predicate="supports",
                object_id="entity_b",
                claim_ids=["claim_1"],
                evidence_ids=["ev_1"],
                confidence=0.7,
            )
        ],
    )

    assert verify_object("relation", "rel_1", tmp_path).valid is True
    result = apply_update_plan(
        {
            "status": "approved",
            "operation": "update_relation",
            "target_type": "relation",
            "target_id": "rel_1",
            "payload": {"change": {"predicate": "affects", "confidence": 0.8}},
        },
        tmp_path,
    )

    relation = JsonlStore(tmp_path).load("edges.jsonl", Relation)[0]
    assert result.status == "applied"
    assert relation.predicate == "affects"
    assert relation.version == 2


def test_changed_source_creates_retire_claim_proposal(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Project Alpha requires Policy Beta.", encoding="utf-8")
    first = ingest_source(source, MockLLMClient(), tmp_path)
    old_claim_id = first.claims[0].id

    source.write_text("Project Alpha requires Policy Gamma.", encoding="utf-8")
    ingest_source(source, MockLLMClient(), tmp_path)

    proposals = JsonlStore(tmp_path).load("proposals.jsonl", UpdateProposalDraft)
    assert old_claim_id in {proposal.target_id for proposal in proposals}
    assert any(proposal.proposal_type == "retire_claim" for proposal in proposals)
