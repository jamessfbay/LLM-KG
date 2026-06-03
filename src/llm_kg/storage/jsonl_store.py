from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _to_json(model: BaseModel) -> str:
    return model.model_dump_json()


def load_jsonl(path: Path, model_type: type[T]) -> list[T]:
    if not path.exists():
        return []
    items: list[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                items.append(model_type.model_validate_json(raw))
            except Exception as exc:  # pragma: no cover - message path is what matters.
                raise ValueError(f"Invalid JSONL record in {path}:{line_no}: {exc}") from exc
    return items


def upsert_jsonl(path: Path, items: list[T]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, BaseModel] = {item.id: item for item in load_jsonl(path, type(items[0]))} if items else {}
    for item in items:
        existing[item.id] = item
    with path.open("w", encoding="utf-8") as handle:
        for item in existing.values():
            handle.write(_to_json(item))
            handle.write("\n")


class JsonlStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.graph_store = workspace / "graph_store"

    def path(self, filename: str) -> Path:
        return self.graph_store / filename

    def load(self, filename: str, model_type: type[T]) -> list[T]:
        return load_jsonl(self.path(filename), model_type)

    def upsert(self, filename: str, items: list[T]) -> None:
        upsert_jsonl(self.path(filename), items)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for filename in (
            "nodes.jsonl",
            "edges.jsonl",
            "claims.jsonl",
            "evidence.jsonl",
            "documents.jsonl",
            "wiki_pages.jsonl",
            "proposals.jsonl",
            "audit_events.jsonl",
            "reasoning_traces.jsonl",
            "cross_validation_runs.jsonl",
        ):
            path = self.path(filename)
            if not path.exists():
                counts[filename] = 0
                continue
            with path.open("r", encoding="utf-8") as handle:
                counts[filename] = sum(1 for line in handle if line.strip())
        return counts
