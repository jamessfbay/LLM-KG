from pathlib import Path

from llm_kg.models import Evidence
from llm_kg.storage import JsonlStore


def test_jsonl_upsert_dedupes_by_id(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path)
    first = Evidence(id="ev_1", source_id="doc_1", quote="Alpha", confidence=0.5)
    second = Evidence(id="ev_1", source_id="doc_1", quote="Beta", confidence=0.8)

    store.upsert("evidence.jsonl", [first])
    store.upsert("evidence.jsonl", [second])

    records = store.load("evidence.jsonl", Evidence)
    assert len(records) == 1
    assert records[0].quote == "Beta"
