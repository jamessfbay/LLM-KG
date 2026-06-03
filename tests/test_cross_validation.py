from pathlib import Path

from llm_kg.models import Claim, CrossValidationClaimResult, CrossValidationProviderResult, Evidence
from llm_kg.storage import JsonlStore
from llm_kg.validation import cross_validate_claims


class FakeValidator:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.model = f"{provider}-test"

    def validate(self, items):
        return CrossValidationProviderResult(
            provider=self.provider,
            model=self.model,
            items=[
                CrossValidationClaimResult(
                    claim_id=item["claim_id"],
                    supported=True,
                    support_level="full",
                    citation_ok=True,
                    planning_fact_ok=True,
                    issues=[],
                    rationale="The quote directly supports the claim.",
                )
                for item in items
            ],
            summary={
                "supported_count": len(items),
                "partial_count": 0,
                "unsupported_count": 0,
                "citation_problem_count": 0,
                "overall_assessment": "All claims are supported.",
            },
        )


def test_cross_validate_claims_builds_consensus_and_persists(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    evidence = Evidence(
        id="ev_1",
        source_id="doc_1",
        quote="Buena Vista Commons contains 61 family units.",
        page_number=2,
        source_mode="native_text",
        confidence=0.99,
    )
    claim = Claim(
        id="claim_1",
        text="Buena Vista Commons contains 61 family units.",
        source_ids=["doc_1"],
        evidence_ids=["ev_1"],
        subject="Buena Vista Commons",
        predicate="contains",
        object="61 family units",
        confidence=0.95,
    )
    store.upsert("evidence.jsonl", [evidence])
    store.upsert("claims.jsonl", [claim])

    result = cross_validate_claims(
        tmp_path,
        providers=["gemini", "xai"],
        validators={"gemini": FakeValidator("gemini"), "xai": FakeValidator("xai")},
        output_path=Path("cross_validation.json"),
    )

    assert result.claim_count == 1
    assert result.consensus[0].verdict == "accepted"
    assert result.consensus[0].supported_by == ["gemini", "xai"]
    assert store.load("cross_validation_runs.jsonl", type(result))[0].id == result.id
    assert (tmp_path / "cross_validation.json").exists()
