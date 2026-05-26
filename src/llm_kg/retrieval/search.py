from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from llm_kg.models import Claim, Evidence, QueryHit
from llm_kg.storage import JsonlStore, MarkdownStore


def search_workspace(question: str, workspace: Path, top_k: int = 5) -> tuple[list[QueryHit], list[Evidence]]:
    terms = _tokenize(question)
    if not terms:
        return [], []
    markdown = MarkdownStore(workspace)
    jsonl = JsonlStore(workspace)
    pages = markdown.load_pages()
    claims = jsonl.load("claims.jsonl", Claim)
    evidence = jsonl.load("evidence.jsonl", Evidence)
    evidence_by_id = {item.id: item for item in evidence}

    docs: list[tuple[str, str, str | None, str | None, list[str]]] = []
    for page in pages:
        docs.append(("wiki", page.id, page.content_md, page.title, page.path, []))
    for claim in claims:
        docs.append(("claim", claim.id, claim.text, None, None, claim.evidence_ids))
    for item in evidence:
        docs.append(("evidence", item.id, item.quote, None, None, [item.id]))

    scored: list[QueryHit] = []
    doc_tokens = [_tokenize(doc[2]) for doc in docs]
    doc_freq = Counter(term for tokens in doc_tokens for term in set(tokens))
    total_docs = max(len(docs), 1)
    for doc, tokens in zip(docs, doc_tokens):
        score = _bm25_like_score(terms, tokens, doc_freq, total_docs)
        if score <= 0:
            continue
        kind, item_id, text, title, path, evidence_ids = doc
        scored.append(
            QueryHit(
                kind=kind,  # type: ignore[arg-type]
                id=item_id,
                title=title,
                text=_snippet(text, terms),
                score=round(score, 4),
                path=path,
                evidence_ids=evidence_ids,
            )
        )
    hits = sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
    cited_evidence = []
    for evidence_id in {eid for hit in hits for eid in hit.evidence_ids}:
        if evidence_id in evidence_by_id:
            cited_evidence.append(evidence_by_id[evidence_id])
    return hits, cited_evidence


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _bm25_like_score(query_terms: list[str], doc_terms: list[str], doc_freq: Counter[str], total_docs: int) -> float:
    counts = Counter(doc_terms)
    length_norm = 1.0 / math.sqrt(max(len(doc_terms), 1))
    score = 0.0
    for term in query_terms:
        if counts[term] == 0:
            continue
        idf = math.log((1 + total_docs) / (1 + doc_freq[term])) + 1
        score += counts[term] * idf
    return score * length_norm


def _snippet(text: str, terms: list[str], width: int = 360) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    start = max(min(positions) - 80, 0) if positions else 0
    snippet = re.sub(r"\s+", " ", text[start : start + width]).strip()
    return snippet
