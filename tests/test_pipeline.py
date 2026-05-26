from pathlib import Path

from llm_kg.api import ingest_source, query_knowledge
from llm_kg.llm.mock_client import MockLLMClient
from llm_kg.pipeline.lint import lint_workspace


def test_ingest_extract_graph_query_and_lint(tmp_path: Path) -> None:
    source = tmp_path / "housing.md"
    source.write_text(
        "SB 330 affects Housing Project Alpha. "
        "Palo Alto Planning Department supports Housing Project Alpha. "
        "Housing Project Alpha requires Traffic Study.",
        encoding="utf-8",
    )

    result = ingest_source(source, llm=MockLLMClient(), workspace=tmp_path)

    assert (tmp_path / result.wiki_page.path).exists()
    assert (tmp_path / "wiki" / "index.md").exists()
    assert (tmp_path / "wiki" / "log.md").exists()
    assert (tmp_path / "graph_store" / "claims.jsonl").exists()
    assert (tmp_path / "graph_store" / "evidence.jsonl").exists()
    assert (tmp_path / "graph_store" / "nodes.jsonl").exists()
    assert (tmp_path / "graph_store" / "edges.jsonl").exists()
    assert result.claims
    assert result.evidence
    assert result.entities

    query = query_knowledge("What affects Housing Project Alpha?", workspace=tmp_path)
    assert query.hits
    assert query.evidence

    issues = lint_workspace(tmp_path)
    assert not [issue for issue in issues if issue.severity == "error"]
