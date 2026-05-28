from __future__ import annotations

from pathlib import Path


class PromptLoader:
    """Load versioned Markdown prompts from the workspace or package defaults."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def load(self, name: str) -> str:
        prompt_path = self.workspace / "config" / "prompts" / f"{name}.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
