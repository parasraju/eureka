import sys

import pytest

sys.path.insert(0, r'D:\p\eureka\src')

from cbf.providers.base import LLMRequest, LLMResponse, generate_request_id
from cbf.storage import BudgetGuard, CallLogger, RunManifest


def test_generate_request_id():
    id1 = generate_request_id()
    id2 = generate_request_id()
    assert id1 != id2
    assert len(id1) == 36


def test_llm_request():
    req = LLMRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
        max_tokens=100,
    )
    assert req.model == "gpt-4o"
    assert req.temperature == 0.7
    assert req.max_tokens == 100


def test_llm_response():
    resp = LLMResponse(
        text="Hello!",
        model="gpt-4o",
        request_id="test-id",
        input_tokens=10,
        output_tokens=5,
        latency_s=1.5,
        raw={"id": "test"},
    )
    assert resp.text == "Hello!"
    assert resp.input_tokens == 10


def test_budget_guard():
    guard = BudgetGuard(max_usd=10.0)
    assert guard.check_and_add(1.0) is True
    assert guard.get_spent() == 1.0
    assert guard.remaining() == 9.0
    assert guard.check_and_add(9.0) is True
    assert guard.check_and_add(1.0) is False
    assert guard.get_spent() == 10.0


def test_call_logger(tmp_path):
    run_id = "test-run"
    logger = CallLogger(run_id, runs_dir=str(tmp_path))

    call_id = logger.log_call(
        provider="openai",
        model="gpt-4o",
        request={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        response={"choices": [{"message": {"content": "hello"}}]},
        latency_s=1.0,
        cost_usd=0.001,
    )

    assert call_id is not None
    calls = logger.load_calls()
    assert len(calls) == 1
    assert calls[0].provider == "openai"
    assert calls[0].cost_usd == 0.001

    manifest = logger.get_manifest()
    assert manifest.total_calls == 1
    assert manifest.total_cost_usd == 0.001

    logger.finalize(status="completed")
    assert manifest.status == "completed"
    assert manifest.ended_utc is not None


def test_run_manifest_serialization():
    manifest = RunManifest(
        run_id="test-123",
        started_utc="2024-01-01T00:00:00Z",
        models=["openai:gpt-4o"],
        total_cost_usd=0.05,
        total_calls=10,
    )
    data = manifest.__dict__
    assert data["run_id"] == "test-123"
    assert data["total_cost_usd"] == 0.05


def test_parse_model_spec():
    from cbf.providers.registry import parse_model_spec

    provider, model = parse_model_spec("openai:gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"

    with pytest.raises(ValueError):
        parse_model_spec("gpt-4o")
