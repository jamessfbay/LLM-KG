import json
from pathlib import Path

from llm_kg.cli import main
from llm_kg.models import Claim, Evidence
from llm_kg.storage import JsonlStore


def test_cli_verify_trace_and_propose(tmp_path: Path, capsys) -> None:
    store = JsonlStore(tmp_path)
    store.upsert("evidence.jsonl", [Evidence(id="ev_1", source_id="doc_1", quote="A requires C.", confidence=0.9)])
    store.upsert(
        "claims.jsonl",
        [Claim(id="claim_1", text="A requires C.", source_ids=["doc_1"], evidence_ids=["ev_1"], confidence=0.9)],
    )
    change_path = tmp_path / "change.json"
    change_path.write_text(json.dumps({"text": "A requires C.", "evidence_ids": ["ev_1"]}), encoding="utf-8")

    assert main(["--workspace", str(tmp_path), "verify", "claim", "claim_1"]) == 0
    assert "valid: True" in capsys.readouterr().out

    assert main(["--workspace", str(tmp_path), "trace", "claim", "claim_1"]) == 0
    assert "claim:claim_1" in capsys.readouterr().out

    assert main(["--workspace", str(tmp_path), "propose", "claim", "claim_1", "--change", str(change_path)]) == 0
    assert "Created proposal" in capsys.readouterr().out
