from datetime import datetime, timezone

from llm_kg.context import build_context_bundle
from llm_kg.models import Claim, Document, Evidence
from llm_kg.storage import JsonlStore


def test_context_admits_only_exact_quote_bound_accepted_facts(tmp_path) -> None:
    document = Document(
        id="doc-1",
        title="Telemetry",
        source_path="https://example.test/telemetry",
        source_type="txt",
        content="Current p95 latency is 120 ms.",
        hash="sha256-content",
    )
    store = JsonlStore(tmp_path)
    store.upsert("documents.jsonl", [document])
    store.upsert(
        "evidence.jsonl",
        [
            Evidence(
                id="ev-good",
                source_id=document.id,
                quote="p95 latency is 120 ms",
                source_content_hash=document.hash,
                quote_start=8,
                quote_end=29,
                confidence=0.8,
                review_state="approved",
            ),
            Evidence(
                id="ev-bad",
                source_id=document.id,
                quote="p95 latency is 20 ms",
                source_content_hash=document.hash,
                quote_start=8,
                quote_end=27,
                confidence=0.99,
                review_state="approved",
            ),
        ],
    )
    store.upsert(
        "claims.jsonl",
        [
            Claim(
                id="claim-good",
                text="Current p95 latency is 120 ms.",
                predicate="p95_latency_ms",
                object="120",
                source_ids=[document.id],
                evidence_ids=["ev-good"],
                confidence=0.7,
                review_state="approved",
            ),
            Claim(
                id="claim-bad",
                text="Current p95 latency is 20 ms.",
                source_ids=[document.id],
                evidence_ids=["ev-bad"],
                confidence=0.99,
                review_state="approved",
            ),
        ],
    )

    bundle = build_context_bundle("current p95 latency", tmp_path)

    assert bundle.generated_answer is None
    assert [fact.id for fact in bundle.facts] == ["claim-good"]
    assert bundle.facts[0].source_content_hash == document.hash
    assert any("claim-bad" in item for item in bundle.unknowns)


def test_context_does_not_hide_conflicting_claim_as_fact(tmp_path) -> None:
    document = Document(id="doc", title="S", source_path="s", source_type="txt", content="capacity is 5", hash="h")
    JsonlStore(tmp_path).upsert("documents.jsonl", [document])
    JsonlStore(tmp_path).upsert("evidence.jsonl", [Evidence(id="ev", source_id="doc", quote="capacity is 5", source_content_hash="h", quote_start=0, quote_end=13, confidence=1, review_state="approved")])
    JsonlStore(tmp_path).upsert("claims.jsonl", [Claim(id="claim", text="capacity is 5", source_ids=["doc"], evidence_ids=["ev"], confidence=1, review_state="approved", conflicts_with=["claim-2"])])

    bundle = build_context_bundle("capacity", tmp_path)

    assert bundle.facts == []
    assert any("unresolved conflicts" in item for item in bundle.unknowns)


def test_context_retrieval_is_not_limited_to_english_tokens(tmp_path) -> None:
    content = "当前服务延迟为 120 毫秒。"
    document = Document(id="doc-cn", title="遥测", source_path="s", source_type="txt", content=content, hash="cn-hash")
    quote = "服务延迟为 120 毫秒"
    start = content.index(quote)
    JsonlStore(tmp_path).upsert("documents.jsonl", [document])
    JsonlStore(tmp_path).upsert("evidence.jsonl", [Evidence(id="ev-cn", source_id=document.id, quote=quote, source_content_hash=document.hash, quote_start=start, quote_end=start + len(quote), confidence=1, review_state="approved")])
    JsonlStore(tmp_path).upsert("claims.jsonl", [Claim(id="claim-cn", text="当前服务延迟为 120 毫秒", source_ids=[document.id], evidence_ids=["ev-cn"], confidence=1, review_state="approved")])

    bundle = build_context_bundle("当前服务延迟", tmp_path)

    assert [fact.id for fact in bundle.facts] == ["claim-cn"]


def test_context_keeps_expired_facts_unknown(tmp_path) -> None:
    content = "capacity is 5"
    document = Document(id="doc", title="S", source_path="s", source_type="txt", content=content, hash="h")
    store = JsonlStore(tmp_path)
    store.upsert("documents.jsonl", [document])
    store.upsert("evidence.jsonl", [Evidence(id="ev", source_id="doc", quote=content, source_content_hash="h", quote_start=0, quote_end=len(content), confidence=1, review_state="approved")])
    store.upsert("claims.jsonl", [Claim(id="expired", text=content, source_ids=["doc"], evidence_ids=["ev"], confidence=1, review_state="approved", valid_to=datetime(2020, 1, 1, tzinfo=timezone.utc))])

    bundle = build_context_bundle("capacity", tmp_path)

    assert bundle.facts == []
    assert any("validity window" in item for item in bundle.unknowns)


def test_context_rejects_evidence_from_an_unrelated_claim_source(tmp_path) -> None:
    document = Document(id="doc", title="S", source_path="s", source_type="txt", content="capacity is 5", hash="h")
    store = JsonlStore(tmp_path)
    store.upsert("documents.jsonl", [document])
    store.upsert("evidence.jsonl", [Evidence(id="ev", source_id="doc", quote=document.content, source_content_hash="h", quote_start=0, quote_end=len(document.content), confidence=1, review_state="approved")])
    store.upsert("claims.jsonl", [Claim(id="claim", text="capacity is 5", source_ids=["different-doc"], evidence_ids=["ev"], confidence=1, review_state="approved")])

    bundle = build_context_bundle("capacity", tmp_path)

    assert bundle.facts == []
    assert any("no immutable exact-quote evidence" in item for item in bundle.unknowns)
