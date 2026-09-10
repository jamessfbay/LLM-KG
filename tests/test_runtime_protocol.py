import json
from io import StringIO
from pathlib import Path

import pytest

from llm_kg.runtime_protocol import (
    OperationReceiptStore,
    RuntimeCommand,
    RuntimeEmitter,
)
from llm_kg.cli import _query_import_scope
from llm_kg.models import Claim, Evidence
from llm_kg.storage import JsonlStore


def test_runtime_fixture_and_ndjson_event(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    stream = StringIO()
    event = RuntimeEmitter(command, stream).emit("step.started", "running", payload={"phase": "test"})
    assert event.protocol_version == "1.0"
    emitted = json.loads(stream.getvalue())
    assert emitted["run_id"] == "run_contract"
    assert emitted["tenant_context"] == {"organization_id": "local", "workspace_id": "default"}


def test_runtime_event_propagates_explicit_tenant_context():
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    payload["tenant_context"] = {"organization_id": "org_1", "workspace_id": "workspace_1", "actor_id": "user_1"}
    command = RuntimeCommand.model_validate(payload)
    stream = StringIO()

    RuntimeEmitter(command, stream).emit("step.started", "running")

    assert json.loads(stream.getvalue())["tenant_context"] == payload["tenant_context"]


def test_operation_receipt_is_idempotent_and_rejects_hash_conflict(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    store = OperationReceiptStore(tmp_path, ".llm_kg")
    receipt = store.begin(command)
    store.complete(receipt, "succeeded", {"artifact_path": "/tmp/result.json"})
    assert store.begin(command).output["artifact_path"] == "/tmp/result.json"

    conflicting = command.model_copy(update={"input_hash": "different"})
    with pytest.raises(ValueError, match="different input hash"):
        store.begin(conflicting)


def test_operation_receipt_serializes_same_idempotency_key(tmp_path: Path):
    payload = json.loads((Path(__file__).parent / "fixtures" / "runtime_command_v1.json").read_text())
    command = RuntimeCommand.model_validate(payload)
    first_store = OperationReceiptStore(tmp_path, ".llm_kg")
    second_store = OperationReceiptStore(tmp_path, ".llm_kg")

    receipt = first_store.begin(command)
    with pytest.raises(RuntimeError, match="already running"):
        second_store.begin(command)

    first_store.complete(receipt, "succeeded", {"artifact_path": "/tmp/result.json"})
    assert second_store.begin(command).status == "succeeded"


def test_runtime_query_is_scoped_to_the_current_import(tmp_path: Path, monkeypatch):
    store = JsonlStore(tmp_path)
    store.upsert("claims.jsonl", [
        Claim(id="claim_current", text="Current imported game AI debugging pain", source_ids=["source_current"], evidence_ids=["ev_current"], confidence=0.9),
        Claim(id="claim_same_source", text="Another claim from the first source", source_ids=["source_current"], evidence_ids=["ev_same_source"], confidence=0.95),
        Claim(id="claim_diverse", text="Independent source corroboration", source_ids=["source_diverse"], evidence_ids=["ev_diverse"], confidence=0.8),
        Claim(id="claim_old", text="Older unrelated market claim", source_ids=["source_old"], evidence_ids=["ev_old"], confidence=0.99),
    ])
    store.upsert("evidence.jsonl", [
        Evidence(id="ev_current", source_id="source_current", quote="Current evidence", confidence=0.9),
        Evidence(id="ev_same_source", source_id="source_current", quote="Same source evidence", confidence=0.95),
        Evidence(id="ev_diverse", source_id="source_diverse", quote="Independent evidence", confidence=0.8),
        Evidence(id="ev_old", source_id="source_old", quote="Old evidence", confidence=0.99),
    ])

    class FakeLLM:
        def answer_question(self, question, context, mode="local"):
            assert "claim_current" in context
            assert "claim_diverse" in context
            assert "claim_same_source" not in context
            assert "claim_old" not in context
            return "Scoped answer"

    monkeypatch.setattr("llm_kg.cli.build_llm_client", lambda settings: FakeLLM())

    result = _query_import_scope(
        "What is relevant?",
        tmp_path,
        top_k=2,
        mode="local",
        claim_ids=["claim_current", "claim_same_source", "claim_diverse"],
        evidence_ids=[],
    )

    assert {hit.id for hit in result.hits} == {"claim_current", "claim_diverse", "ev_current", "ev_diverse"}
    assert result.trace_id
