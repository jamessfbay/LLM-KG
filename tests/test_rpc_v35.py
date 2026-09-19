from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_kg.rpc import ENGINE_ID, PROTOCOL_SCHEMA_HASH, EngineRpcRequest, RpcError, invoke


def rpc_request(**command_overrides):
    command = {
        "protocol_version": "3.5", "command_id": "cmd-1", "run_id": "run-1",
        "adapter_id": "kg", "capability": "context_provider", "operation": "context",
        "input": {"objective": "missing fact"}, "input_hash": "b" * 64,
        "idempotency_key": "context-once", "fencing": {}, "lease_token": 2,
        "deadline": int(datetime.now(UTC).timestamp() * 1000) + 60_000,
    }
    command.update(command_overrides)
    return EngineRpcRequest(protocol_version="3.5", protocol_schema_hash=PROTOCOL_SCHEMA_HASH, engine=ENGINE_ID, command=command)


def test_context_rpc_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_KG_PROVIDER", "mock")
    first = invoke(rpc_request(), tmp_path)
    second = invoke(rpc_request(command_id="cmd-retry", lease_token=9), tmp_path)
    assert first.reply.protocol_version == "3.5"
    assert first.replayed is False
    assert second.replayed is True
    assert second.reply.output == first.reply.output
    assert second.reply.command_id == "cmd-retry"
    assert second.reply.lease_token == 9


def test_context_rpc_rejects_schema_and_idempotency_conflicts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_KG_PROVIDER", "mock")
    request = rpc_request()
    request.protocol_schema_hash = "0" * 64
    with pytest.raises(RpcError, match="schema hash"):
        invoke(request, tmp_path)
    invoke(rpc_request(), tmp_path)
    with pytest.raises(RpcError, match="another input hash"):
        invoke(rpc_request(input_hash="c" * 64), tmp_path)
