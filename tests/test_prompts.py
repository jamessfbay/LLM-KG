from pathlib import Path

import pytest

from llm_kg.prompts import PromptLoader


def test_prompt_loader_reads_workspace_prompt(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "config" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "answer_basic.md").write_text("# Answer Basic\n\nUse evidence.", encoding="utf-8")

    prompt = PromptLoader(tmp_path).load("answer_basic")

    assert "Use evidence" in prompt


def test_prompt_loader_errors_on_missing_prompt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        PromptLoader(tmp_path).load("missing")
