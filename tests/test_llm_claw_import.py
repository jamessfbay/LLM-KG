import json
from pathlib import Path

from llm_kg.api import ingest_source
from llm_kg.cli import main
from llm_kg.models import Claim, Document, Evidence
from llm_kg.storage import JsonlStore


def _payload() -> dict:
    return {
        "format": "llm-kg-import",
        "request_id": "task_demo",
        "documents": [
            {
                "id": "doc_demo",
                "title": "Demo Staff Report",
                "source_path": "https://example.test/staff-report",
                "source_type": "md",
                "content": "The project has an appeal filing.",
                "hash": "hash_demo",
                "metadata": {"source_url": "https://example.test/staff-report"},
            }
        ],
        "evidence": [
            {
                "id": "ev_demo",
                "source_id": "doc_demo",
                "quote": "The project has an appeal filing.",
                "url": "https://example.test/staff-report",
                "source_mode": "native_text",
                "confidence": 0.91,
                "review_state": "auto_accepted",
            }
        ],
        "claims": [
            {
                "id": "claim_demo",
                "text": "The project has an appeal filing.",
                "source_ids": ["doc_demo"],
                "evidence_ids": ["ev_demo"],
                "confidence": 0.91,
                "status": "active",
                "review_state": "auto_accepted",
            }
        ],
        "missing_data": ["permit history"],
        "recommended_next_actions": ["review appeal packet"],
    }


def test_ingest_llm_claw_import_json_writes_graph_records(tmp_path: Path) -> None:
    source = tmp_path / "import.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    result = ingest_source(source, workspace=tmp_path)
    store = JsonlStore(tmp_path)

    assert result.ingest_status == "ingested"
    assert result.source_format == "llm-kg-import"
    assert result.request_id == "task_demo"
    assert result.document.title == "LLM-CLAW Evidence Import task_demo"
    assert result.claims[0].id == "claim_demo"
    assert result.evidence[0].id == "ev_demo"
    assert store.load("documents.jsonl", Document)
    assert store.load("claims.jsonl", Claim)[0].id == "claim_demo"
    assert store.load("evidence.jsonl", Evidence)[0].id == "ev_demo"
    assert (tmp_path / result.wiki_page.path).exists()


def test_cli_ingest_llm_claw_import_json(tmp_path: Path, capsys) -> None:
    source = tmp_path / "import.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    assert main(["--workspace", str(tmp_path), "--json", "ingest", str(source)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["claims"][0]["id"] == "claim_demo"
    assert output["evidence"][0]["id"] == "ev_demo"
    assert output["ingest_status"] == "ingested"


def test_empty_claw_import_is_explicitly_insufficient(tmp_path: Path) -> None:
    source = tmp_path / "empty_import.json"
    source.write_text(
        json.dumps(
            {
                "format": "llm-kg-import",
                "request_id": "task_empty",
                "documents": [],
                "claims": [],
                "evidence": [],
                "missing_data": ["No verified raw-source evidence found."],
            }
        ),
        encoding="utf-8",
    )

    result = ingest_source(source, workspace=tmp_path)

    assert result.ingest_status == "empty_import"
    assert result.request_id == "task_empty"
    assert result.claims == []
    assert result.evidence == []
