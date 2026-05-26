from pathlib import Path

from llm_kg.models import Claim, Entity, Relation
from llm_kg.pipeline.lint import lint_workspace
from llm_kg.storage import JsonlStore


def test_lint_reports_bad_refs_and_missing_evidence(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki" / "sources"
    wiki.mkdir(parents=True)
    (wiki / "source.md").write_text("# Source\n\n[[Missing Entity]]", encoding="utf-8")

    store = JsonlStore(tmp_path)
    store.upsert("claims.jsonl", [Claim(id="claim_1", text="Unsupported", confidence=0.5)])
    store.upsert("nodes.jsonl", [Entity(id="entity_1", name="Alpha", entity_type="concept")])
    store.upsert(
        "edges.jsonl",
        [
            Relation(
                id="rel_1",
                subject_id="entity_1",
                predicate="mentions",
                object_id="missing",
                claim_ids=["missing_claim"],
                evidence_ids=["missing_evidence"],
                confidence=0.5,
            )
        ],
    )

    codes = {issue.code for issue in lint_workspace(tmp_path)}

    assert "broken_wikilink" in codes
    assert "claim_missing_evidence" in codes
    assert "relation_bad_object_ref" in codes
    assert "relation_bad_claim_ref" in codes
    assert "relation_bad_evidence_ref" in codes
