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


def test_estimate_run_cost_deterministic():
    from cbf.runner.estimate import estimate_run_cost

    config = {"benchmarks": {}, "conditions": {}}
    conditions = {"cell": {"condition": "neutral", "num_samples": 2}}
    first = estimate_run_cost(config, "tier_a", conditions, "openai:gpt-4o")
    second = estimate_run_cost(config, "tier_a", conditions, "openai:gpt-4o")

    assert first == second
    assert first["calls"] == 2 + 20 + 50
    assert first["total_usd"] > 0
    assert set(first["breakdown"]) == {"dat", "aut", "diversity"}
    assert all(entry["calls"] > 0 for entry in first["breakdown"].values())
    assert first["assumptions"]


def test_human_norms_missing_and_malformed(tmp_path):
    from cbf.metrics.human_norms import HumanNormUnavailable, load_norm, percentile

    with pytest.raises(HumanNormUnavailable, match="No human-norm baseline"):
        percentile(50.0, "dat", tmp_path)

    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(HumanNormUnavailable, match="Could not read"):
        load_norm("bad", tmp_path)

    (tmp_path / "incomplete.json").write_text('{"instrument": "incomplete"}', encoding="utf-8")
    with pytest.raises(HumanNormUnavailable, match="missing required key"):
        load_norm("incomplete", tmp_path)

    (tmp_path / "zerosd.json").write_text(
        '{"instrument": "zerosd", "source_url": "https://example.org", '
        '"license": "CC-BY-4.0", "retrieval_date": "2026-01-01", "n": 10, '
        '"mean": 50, "sd": 0}',
        encoding="utf-8",
    )
    with pytest.raises(HumanNormUnavailable, match="non-positive sd"):
        load_norm("zerosd", tmp_path)


def test_human_norms_percentile_from_sourced_file(tmp_path):
    from cbf.metrics.human_norms import percentile

    (tmp_path / "dat.json").write_text(
        '{"instrument": "dat", "source_url": "https://example.org/norms", '
        '"license": "CC-BY-4.0", "retrieval_date": "2026-01-01", "n": 120, '
        '"mean": 70, "sd": 10, "higher_is_better": true}',
        encoding="utf-8",
    )

    assert percentile(70, "dat", tmp_path) == pytest.approx(50.0)
    assert percentile(80, "dat", tmp_path) == pytest.approx(84.134, abs=0.01)
    assert percentile(-1000, "dat", tmp_path) == 0.0
    assert percentile(1000, "dat", tmp_path) == 100.0
    assert percentile(80, "dat", tmp_path, higher_is_better=False) == pytest.approx(15.866, abs=0.01)
