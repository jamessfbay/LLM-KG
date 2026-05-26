from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_kg.api import ingest_source, lint_workspace, query_knowledge
from llm_kg.config import Settings
from llm_kg.storage import JsonlStore, MarkdownStore


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

    subparsers.add_parser("lint", help="Lint wiki and graph records")
    subparsers.add_parser("stats", help="Show workspace stats")

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
        result = query_knowledge(args.question, workspace=workspace, top_k=top_k)
        text = [result.answer, ""]
        for hit in result.hits:
            label = f"{hit.kind}:{hit.id}"
            text.append(f"- [{label}] score={hit.score} {hit.title or ''} {hit.path or ''}".strip())
            text.append(f"  {hit.text}")
        return _print(args.json, result, "\n".join(text).strip())
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
        return 0
    return 2


def _print(as_json: bool, model: BaseModel, text: str) -> int:
    if as_json:
        print(model.model_dump_json(indent=2))
    else:
        print(text)
    return 0


def _stats(workspace: Path) -> dict[str, Any]:
    return {
        "workspace": str(workspace),
        "wiki_pages": len(MarkdownStore(workspace).load_pages()),
        "graph_records": JsonlStore(workspace).counts(),
    }
