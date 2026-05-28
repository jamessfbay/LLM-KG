from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.embeddings import build_embedding_client
from llm_kg.llm import LLMClient
from llm_kg.models import QueryResult
from llm_kg.retrieval import search_workspace
from llm_kg.storage import build_postgres_store


def query_knowledge(
    question: str,
    llm: LLMClient,
    workspace: Path,
    top_k: int = 5,
    mode: str = "local",
) -> QueryResult:
    if mode not in {"basic", "local"}:
        raise ValueError("mode must be one of: basic, local")
    settings = Settings.from_env(workspace)
    postgres = build_postgres_store(settings)
    if postgres:
        embedder = build_embedding_client(settings)
        query_embedding = embedder.embed_text(question)
        if mode == "basic":
            hits = postgres.search_basic(question, query_embedding, top_k=top_k)
            evidence = []
        else:
            hits, evidence = postgres.search_local(question, query_embedding, top_k=top_k)
    else:
        hits, evidence = search_workspace(question, workspace, top_k=top_k)
    context = "\n\n".join(
        f"[{hit.kind}:{hit.id}] {hit.title or ''}\n{hit.text}\nEvidence: {', '.join(hit.evidence_ids)}"
        for hit in hits
    )
    answer = llm.answer_question(question, context, mode=mode)
    return QueryResult(question=question, mode=mode, answer=answer, hits=hits, evidence=evidence)
