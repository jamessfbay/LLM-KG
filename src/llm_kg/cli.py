from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_kg.api import (
    apply_update_plan,
    create_proposal,
    export_reasoning_trace,
    export_proposal,
    ingest_source,
    list_reasoning_traces,
    lint_workspace,
    query_knowledge,
    trace_object,
    trace_query,
    verify_claim,
    verify_object,
)
from llm_kg.config import Settings
from llm_kg.models import Claim, Document, Entity, Evidence, Relation
from llm_kg.storage import JsonlStore, MarkdownStore, build_postgres_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-kg", description="LLM-KG Markdown + JSONL MVP")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a source file")
    ingest_parser.add_argument("path", type=Path)

    query_parser = subparsers.add_parser("query", help="Query wiki and graph")
    query_parser.add_argument("question")
    query_parser.add_argument("--top-k", type=int, default=None)
    query_parser.add_argument("--mode", choices=["basic", "local"], default=None)

    subparsers.add_parser("lint", help="Lint wiki and graph records")
    subparsers.add_parser("stats", help="Show workspace stats")

    verify_parser = subparsers.add_parser("verify", help="Verify evidence governance for an object")
    verify_subparsers = verify_parser.add_subparsers(dest="verify_type", required=True)
    for verify_type in ("claim", "relation", "entity", "evidence", "wiki_page"):
        verify_item_parser = verify_subparsers.add_parser(verify_type, help=f"Verify a {verify_type}")
        verify_item_parser.add_argument("target_id")

    trace_parser = subparsers.add_parser("trace", help="Trace an object's evidence path")
    trace_parser.add_argument("target_type", choices=["claim", "relation", "entity", "evidence", "query"])
    trace_parser.add_argument("target_id")

    traces_parser = subparsers.add_parser("traces", help="List or export reasoning traces")
    traces_subparsers = traces_parser.add_subparsers(dest="traces_command", required=True)
    traces_subparsers.add_parser("list", help="List reasoning traces")
    traces_export_parser = traces_subparsers.add_parser("export", help="Export a reasoning trace")
    traces_export_parser.add_argument("trace_id")
    traces_export_parser.add_argument("--format", default="llm-kee", choices=["llm-kee"])

    propose_parser = subparsers.add_parser("propose", help="Create an LLM-KEE-compatible update proposal")
    propose_parser.add_argument("target_type", choices=["claim", "relation", "entity", "evidence", "wiki_page"])
    propose_parser.add_argument("target_id")
    propose_parser.add_argument("--change", required=True, help="Path to JSON change payload")

    export_parser = subparsers.add_parser("export-proposal", help="Export a proposal for LLM-KEE")
    export_parser.add_argument("proposal_id")
    export_parser.add_argument("--format", default="llm-kee", choices=["llm-kee"])

    apply_parser = subparsers.add_parser("apply-plan", help="Apply an approved LLM-KEE update plan")
    apply_parser.add_argument("plan_json", help="Path to approved UpdatePlan JSON")

    db_parser = subparsers.add_parser("db", help="Manage PostgreSQL+pgvector storage")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)
    db_subparsers.add_parser("init", help="Apply database migrations")
    db_subparsers.add_parser("migrate", help="Apply database migrations")
    db_subparsers.add_parser("status", help="Show database status")

    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()

    if args.command == "ingest":
        result = ingest_source(args.path, workspace=workspace)
        return _print(
            args.json,
            result,
            f"Ingested {result.document.title}: {len(result.claims)} claims, "
            f"{len(result.entities)} entities, {len(result.relations)} relations\n"
            f"Wiki page: {result.wiki_page.path}",
        )
    if args.command == "query":
        settings = Settings.from_env(workspace)
        top_k = args.top_k or settings.top_k
        mode = args.mode or settings.query_default_mode
        result = query_knowledge(args.question, workspace=workspace, top_k=top_k, mode=mode)
        text = [result.answer, ""]
        for hit in result.hits:
            label = f"{hit.kind}:{hit.id}"
            text.append(f"- [{label}] score={hit.score} {hit.title or ''} {hit.path or ''}".strip())
            text.append(f"  {hit.text}")
        return _print(args.json, result, "\n".join(text).strip())
    if args.command == "db":
        settings = Settings.from_env(workspace)
        postgres = build_postgres_store(settings)
        if postgres is None:
            print("LLM_KG_DATABASE_URL or [database].url is required for db commands.")
            return 2
        if args.db_command in {"init", "migrate"}:
            postgres.apply_migrations(workspace / "migrations")
            print("Database migrations applied.")
            return 0
        if args.db_command == "status":
            status = postgres.status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"Database: {status['database']}")
                print(f"User: {status['user']}")
                print(f"Migrated: {status['migrated']}")
                print(f"Documents: {status['documents']}")
                print(f"Embeddings: {status['embeddings']}")
                print(f"Audit events: {status.get('audit_events', 0)}")
                print(f"Proposals: {status.get('proposals', 0)}")
                print(f"Reasoning traces: {status.get('reasoning_traces', 0)}")
            return 0
    if args.command == "verify":
        result = verify_claim(args.target_id, workspace=workspace) if args.verify_type == "claim" else verify_object(args.verify_type, args.target_id, workspace=workspace)
        return _print(
            args.json,
            result,
            _human_verification(result),
        )
    if args.command == "trace":
        if args.target_type == "query":
            result = trace_query(args.target_id, workspace=workspace)
            return _print(args.json, result, _human_query_trace(result))
        result = trace_object(args.target_type, args.target_id, workspace=workspace)
        return _print(args.json, result, _human_object_trace(result))
    if args.command == "traces":
        if args.traces_command == "list":
            traces = list_reasoning_traces(workspace=workspace)
            if args.json:
                print(json.dumps([trace.model_dump(mode="json") for trace in traces], indent=2))
            else:
                for trace in traces:
                    print(f"{trace.id} {trace.mode} confidence={trace.confidence} {trace.question}")
            return 0
        payload = export_reasoning_trace(args.trace_id, workspace=workspace, export_format=args.format)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "propose":
        change = read_json(args.change)
        proposal = create_proposal(args.target_type, args.target_id, change, workspace=workspace)
        return _print(args.json, proposal, f"Created proposal {proposal.id} for {proposal.target_type} {proposal.target_id}")
    if args.command == "export-proposal":
        payload = export_proposal(args.proposal_id, workspace=workspace, export_format=args.format)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "apply-plan":
        result = apply_update_plan(read_json(args.plan_json), workspace=workspace)
        return _print(args.json, result, result.message)
    if args.command == "lint":
        issues = lint_workspace(workspace=workspace)
        if args.json:
            print(json.dumps([issue.model_dump(mode="json") for issue in issues], indent=2))
        elif issues:
            for issue in issues:
                where = f" ({issue.path})" if issue.path else ""
                print(f"{issue.severity.upper()} {issue.code}{where}: {issue.message}")
        else:
            print("No lint issues found.")
        return 1 if any(issue.severity == "error" for issue in issues) else 0
    if args.command == "stats":
        stats = _stats(workspace)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Wiki pages: {stats['wiki_pages']}")
            for name, count in stats["graph_records"].items():
                print(f"{name}: {count}")
            quality = stats.get("extraction_quality", {})
            if quality:
                print("Extraction quality:")
                for key, value in quality.items():
                    print(f"  {key}: {value}")
        return 0
    return 2


def _print(as_json: bool, model: BaseModel, text: str) -> int:
    if as_json:
        print(model.model_dump_json(indent=2))
    else:
        print(text)
    return 0


def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _human_verification(result) -> str:
    lines = [f"Verification for {result.target_type}:{result.target_id}", f"valid: {result.valid}"]
    if result.review_state:
        lines.append(f"review_state: {result.review_state}")
    if result.confidence is not None:
        lines.append(f"confidence: {result.confidence}")
    if result.evidence:
        lines.append(f"evidence: {', '.join(item.id for item in result.evidence)}")
    for issue in result.issues:
        lines.append(f"{issue.severity.upper()} {issue.code}: {issue.message}")
    return "\n".join(lines)


def _human_object_trace(result) -> str:
    lines = [f"Trace for {result.target_type}:{result.target_id}"]
    for node in result.nodes:
        title = f" {node.title}" if node.title else ""
        lines.append(f"- {node.kind}:{node.id}{title}")
        if node.text:
            lines.append(f"  {node.text[:240]}")
    for gap in result.gaps:
        lines.append(f"{gap.severity.upper()} {gap.code}: {gap.message}")
    return "\n".join(lines)


def _human_query_trace(result) -> str:
    lines = [f"Query trace {result.id}", f"mode: {result.mode}", f"confidence: {result.confidence}", result.question, ""]
    lines.append(result.answer)
    if result.used_claim_ids:
        lines.append(f"claims: {', '.join(result.used_claim_ids)}")
    if result.used_relation_ids:
        lines.append(f"relations: {', '.join(result.used_relation_ids)}")
    if result.used_evidence_ids:
        lines.append(f"evidence: {', '.join(result.used_evidence_ids)}")
    return "\n".join(lines).strip()


def _stats(workspace: Path) -> dict[str, Any]:
    settings = Settings.from_env(workspace)
    jsonl = JsonlStore(workspace)
    stats = {
        "workspace": str(workspace),
        "wiki_pages": len(MarkdownStore(workspace).load_pages()),
        "graph_records": jsonl.counts(),
        "extraction_quality": _extraction_quality(jsonl),
    }
    postgres = build_postgres_store(settings)
    if postgres:
        try:
            stats["postgres"] = postgres.status()
        except Exception as exc:
            stats["postgres"] = {"error": str(exc)}
    return stats


def _extraction_quality(jsonl: JsonlStore) -> dict[str, Any]:
    documents = jsonl.load("documents.jsonl", Document)
    claims = jsonl.load("claims.jsonl", Claim)
    evidence = jsonl.load("evidence.jsonl", Evidence)
    entities = jsonl.load("nodes.jsonl", Entity)
    relations = jsonl.load("edges.jsonl", Relation)
    evidence_ids = {item.id for item in evidence}
    entity_ids = {item.id for item in entities}
    pdf_coverages = [
        document.metadata.get("pdf_page_coverage")
        for document in documents
        if document.source_type == "pdf" and isinstance(document.metadata.get("pdf_page_coverage"), dict)
    ]
    noisy_entities = [
        entity
        for entity in entities
        if "\n" in entity.name or len(entity.name) > 80 or (entity.name.isupper() and len(entity.name.split()) > 4)
    ]
    valid_relations = [
        relation
        for relation in relations
        if relation.subject_id in entity_ids
        and relation.object_id in entity_ids
        and all(evidence_id in evidence_ids for evidence_id in relation.evidence_ids)
    ]
    quality: dict[str, Any] = {
        "claims_with_evidence": f"{sum(1 for claim in claims if claim.evidence_ids)}/{len(claims)}",
        "claims_with_page_numbers": f"{sum(1 for claim in claims if _claim_has_page(claim, evidence))}/{len(claims)}",
        "noisy_entity_ratio": round(len(noisy_entities) / len(entities), 3) if entities else 0,
        "relation_validity": f"{len(valid_relations)}/{len(relations)}",
        "ocr_evidence_count": sum(1 for item in evidence if item.source_mode == "ocr_text"),
    }
    if pdf_coverages:
        quality["pdf_page_coverage"] = {
            "total_pages": sum(int(item.get("total_pages") or 0) for item in pdf_coverages),
            "text_pages": sum(int(item.get("text_pages") or 0) for item in pdf_coverages),
            "native_pages": sum(int(item.get("native_pages") or 0) for item in pdf_coverages),
            "ocr_pages": sum(int(item.get("ocr_pages") or 0) for item in pdf_coverages),
            "timeout_pages": sum(int(item.get("timeout_pages") or 0) for item in pdf_coverages),
            "failed_pages": sum(int(item.get("failed_pages") or 0) for item in pdf_coverages),
        }
    return quality


def _claim_has_page(claim: Claim, evidence: list[Evidence]) -> bool:
    by_id = {item.id: item for item in evidence}
    return any(by_id.get(evidence_id) and by_id[evidence_id].page_number is not None for evidence_id in claim.evidence_ids)
