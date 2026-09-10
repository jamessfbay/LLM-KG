from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from llm_kg.models import Claim, ContextBundle, ContextFact, Document, Evidence
from llm_kg.storage import JsonlStore


def build_context_bundle(question: str, workspace: Path, top_k: int = 5) -> ContextBundle:
    """Build decision context from governed records without generating an answer.

    A fact is admitted only when an accepted claim has an exact quotation bound to
    an immutable source hash. Retrieval rank is never represented as probability.
    """
    store = JsonlStore(workspace.resolve())
    documents = {item.id: item for item in store.load("documents.jsonl", Document)}
    evidence = {item.id: item for item in store.load("evidence.jsonl", Evidence)}
    claims = store.load("claims.jsonl", Claim)
    captured_at = datetime.now(timezone.utc)
    query_terms = _terms(question)
    ranked = sorted(
        claims,
        key=lambda claim: (-_overlap(query_terms, _terms(_claim_search_text(claim))), claim.id),
    )

    facts: list[ContextFact] = []
    unknowns: list[str] = []
    for claim in ranked:
        if len(facts) >= max(0, top_k):
            break
        if query_terms and _overlap(query_terms, _terms(_claim_search_text(claim))) == 0:
            continue
        if claim.review_state not in {"approved", "auto_accepted"} or claim.status != "active":
            unknowns.append(f"Claim {claim.id} is not an accepted active fact")
            continue
        if claim.conflicts_with:
            unknowns.append(f"Claim {claim.id} has unresolved conflicts")
            continue
        if (claim.valid_from and claim.valid_from > captured_at) or (
            claim.valid_to and claim.valid_to <= captured_at
        ):
            unknowns.append(f"Claim {claim.id} is outside its validity window")
            continue
        bound = next(
            (
                binding
                for evidence_id in claim.evidence_ids
                if (item := evidence.get(evidence_id)) is not None
                and item.source_id in claim.source_ids
                and (binding := _bound_evidence(item, documents)) is not None
            ),
            None,
        )
        if bound is None:
            unknowns.append(f"Claim {claim.id} has no immutable exact-quote evidence")
            continue
        item, source_hash, observed_at = bound
        facts.append(
            ContextFact(
                id=claim.id,
                key=_fact_key(claim),
                value={
                    "text": claim.text,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                },
                source_id=item.source_id,
                source_content_hash=source_hash,
                source_version=item.version,
                evidence_ids=[item.id],
                observed_at=claim.observed_at or observed_at,
                valid_from=claim.valid_from,
                valid_to=claim.valid_to,
                conflicts_with=claim.conflicts_with,
            )
        )

    if not facts:
        unknowns.append("No accepted fact with immutable exact-quote evidence matched the query")
    return ContextBundle(
        query=question,
        captured_at=captured_at,
        facts=facts,
        unknowns=list(dict.fromkeys(unknowns)),
    )


def _bound_evidence(
    evidence: Evidence, documents: dict[str, Document]
) -> tuple[Evidence, str, object] | None:
    if evidence.review_state not in {"approved", "auto_accepted"} or not evidence.quote.strip():
        return None
    document = documents.get(evidence.source_id)
    if document is None or not evidence.source_content_hash or evidence.source_content_hash != document.hash:
        return None
    if evidence.quote_start is None or evidence.quote_end is None or evidence.quote_end <= evidence.quote_start:
        return None
    if document.content[evidence.quote_start:evidence.quote_end] != evidence.quote:
        return None
    return evidence, document.hash, evidence.observed_at or document.ingested_at


def _terms(value: str) -> set[str]:
    normalized = value.lower()
    terms = {term for term in re.findall(r"[a-z0-9]+", normalized) if len(term) >= 3}
    for run in re.findall(r"[\u3400-\u9fff]+", normalized):
        terms.update(run[index:index + 2] for index in range(max(0, len(run) - 1)))
    return terms


def _overlap(left: set[str], right: set[str]) -> int:
    return len(left & right)


def _claim_search_text(claim: Claim) -> str:
    return " ".join(str(value or "") for value in (claim.text, claim.subject, claim.predicate, claim.object))


def _fact_key(claim: Claim) -> str:
    if claim.subject and claim.predicate:
        return f"{claim.subject}.{claim.predicate}"
    return claim.id
