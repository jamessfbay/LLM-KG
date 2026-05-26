from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from llm_kg.models import WikiPage

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


class MarkdownStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.wiki_dir = workspace / "wiki"

    def save_page(self, page: WikiPage) -> None:
        path = self.workspace / page.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.content_md, encoding="utf-8")

    def load_pages(self) -> list[WikiPage]:
        pages: list[WikiPage] = []
        if not self.wiki_dir.exists():
            return pages
        for path in sorted(self.wiki_dir.rglob("*.md")):
            if path.name in {"index.md", "log.md"}:
                continue
            content = path.read_text(encoding="utf-8")
            page_type = path.parent.name.rstrip("s")
            if page_type not in {"source", "entity", "concept", "synthesis", "comparison"}:
                page_type = "source"
            pages.append(
                WikiPage(
                    id=path.stem,
                    title=_title_from_content(content, path.stem),
                    page_type=page_type,  # type: ignore[arg-type]
                    path=str(path.relative_to(self.workspace)),
                    content_md=content,
                    source_ids=_source_ids_from_content(content),
                    wikilinks=WIKILINK_RE.findall(content),
                    tags=[],
                    updated_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
                )
            )
        return pages

    def update_index(self) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        pages = self.load_pages()
        lines = ["# LLM-KG Index", ""]
        for page in sorted(pages, key=lambda p: (p.page_type, p.title.lower())):
            lines.append(f"- [{page.title}]({Path(page.path).as_posix()}) `{page.page_type}`")
        (self.wiki_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def append_log(self, message: str) -> None:
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.wiki_dir / "log.md"
        if not log_path.exists():
            log_path.write_text("# LLM-KG Log\n\n", encoding="utf-8")
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {timestamp} {message}\n")


def _title_from_content(content: str, fallback: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback.replace("-", " ").replace("_", " ").title()


def _source_ids_from_content(content: str) -> list[str]:
    match = re.search(r"source_id:\s*`?([A-Za-z0-9_.:-]+)`?", content)
    return [match.group(1)] if match else []
