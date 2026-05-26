from __future__ import annotations

from pathlib import Path

from llm_kg.llm import LLMClient
from llm_kg.models import QueryResult
from llm_kg.retrieval import search_workspace


def query_knowledge(question: str, llm: LLMClient, workspace: Path, top_k: int = 5) -> QueryResult:
    hits, evidence = search_workspace(question, workspace, top_k=top_k)
    context = "\n\n".join(
        f"[{hit.kind}:{hit.id}] {hit.title or ''}\n{hit.text}\nEvidence: {', '.join(hit.evidence_ids)}"
        for hit in hits
    )
    answer = llm.answer_question(question, context)
    return QueryResult(question=question, answer=answer, hits=hits, evidence=evidence)
