from __future__ import annotations

from pathlib import Path

from llm_kg.config import Settings
from llm_kg.models import QueryHit, ReasoningStep, ReasoningTrace
from llm_kg.readers.common import stable_id
from llm_kg.storage import JsonlStore, build_postgres_store


def build_reasoning_trace(question: str, answer: str, mode: str, hits: list[QueryHit]) -> ReasoningTrace:
    claim_ids = [hit.id for hit in hits if hit.kind == "claim"]
    relation_ids = [hit.id for hit in hits if hit.kind == "relation"]
    evidence_ids = sorted({evidence_id for hit in hits for evidence_id in hit.evidence_ids} | {hit.id for hit in hits if hit.kind == "evidence"})
    step = ReasoningStep(
        id=stable_id("step", f"{question}:{mode}:retrieval"),
        order=1,
        description=f"Retrieved {len(hits)} {mode} context hits and assembled an evidence-grounded answer draft.",
        claim_ids=claim_ids,
        relation_ids=relation_ids,
        evidence_ids=evidence_ids,
    )
    confidence = min(0.95, max(0.1, sum(hit.score for hit in hits[:5]) / max(len(hits[:5]), 1)))
    return ReasoningTrace(
        id=stable_id("trace", f"{question}:{answer}:{mode}:{','.join(hit.id for hit in hits)}"),
        question=question,
        answer=answer,
        mode=mode,  # type: ignore[arg-type]
        hits=hits,
        used_claim_ids=claim_ids,
        used_relation_ids=relation_ids,
        used_evidence_ids=evidence_ids,
        reasoning_steps=[step],
        confidence=round(confidence, 4),
        decision_output=answer,
    )


def save_reasoning_trace(trace: ReasoningTrace, workspace: Path) -> ReasoningTrace:
    JsonlStore(workspace).upsert("reasoning_traces.jsonl", [trace])
    postgres = build_postgres_store(Settings.from_env(workspace))
    if postgres:
        postgres.upsert_reasoning_trace(trace)
    return trace


def get_reasoning_trace(trace_id: str, workspace: Path) -> ReasoningTrace | None:
    postgres = build_postgres_store(Settings.from_env(workspace))
    if postgres:
        trace = postgres.get_reasoning_trace(trace_id)
        if trace:
            return trace
    return next((trace for trace in JsonlStore(workspace).load("reasoning_traces.jsonl", ReasoningTrace) if trace.id == trace_id), None)


def list_reasoning_traces(workspace: Path) -> list[ReasoningTrace]:
    return JsonlStore(workspace).load("reasoning_traces.jsonl", ReasoningTrace)


def export_reasoning_trace(trace_id: str, workspace: Path, export_format: str = "llm-kee") -> dict:
    if export_format != "llm-kee":
        raise ValueError("Only llm-kee export format is supported.")
    trace = get_reasoning_trace(trace_id, workspace)
    if not trace:
        raise ValueError(f"Reasoning trace not found: {trace_id}")
    return {
        "id": trace.id,
        "question": trace.question,
        "final_answer": trace.answer,
        "reasoning_steps": [
            {
                "id": step.id,
                "order": step.order,
                "description": step.description,
                "claim_ids": step.claim_ids,
                "evidence_ids": step.evidence_ids,
            }
            for step in trace.reasoning_steps
        ],
        "used_claim_ids": trace.used_claim_ids,
        "used_relation_ids": trace.used_relation_ids,
        "used_evidence_ids": trace.used_evidence_ids,
        "confidence": trace.confidence,
        "reusable": False,
    }
