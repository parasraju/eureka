import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cbf.benchmarks import dat as dat_module  # noqa: E402
from cbf.benchmarks.base import BenchmarkItem, RawResults  # noqa: E402
from cbf.benchmarks.dat import DATBenchmark  # noqa: E402
from cbf.providers import LLMRequest, LLMResponse, ProviderRegistry  # noqa: E402
from cbf.runner import BenchmarkRunner, run_benchmarks  # noqa: E402
from cbf.storage import (  # noqa: E402
    BudgetExceededError,
    BudgetGuard,
    CallLogger,
    load_calls,
    load_manifest,
)

DAT_WORDS = [
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "honeydew",
    "kiwi",
    "lemon",
]
DAT_TEXT = ", ".join(DAT_WORDS)
DAT_VOCAB = {word: np.eye(len(DAT_WORDS))[idx] for idx, word in enumerate(DAT_WORDS)}


class FakeProvider:
    name = "fake"

    def __init__(self, cost: float = 0.0, text: str = DAT_TEXT, error: str | None = None):
        self.cost = cost
        self.text = text
        self.error = error
        self.calls = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.error is not None:
            raise RuntimeError(self.error)
        return LLMResponse(
            text=self.text,
            model=req.model,
            request_id="provider-request-id",
            input_tokens=10,
            output_tokens=5,
            latency_s=0.01,
            raw={"id": "provider-request-id"},
        )

    def cost_usd(self, resp: LLMResponse) -> float:
        return self.cost


def patch_initialize(monkeypatch, tmp_path, provider):
    async def fake_initialize(self, model_spec, max_usd=None):
        self.registry = ProviderRegistry()
        self.run_id = "test-run"
        self.logger = CallLogger(self.run_id, runs_dir=str(tmp_path))
        self.budget = BudgetGuard(max_usd if max_usd is not None else 100.0)
        self.logger.update_manifest(
            models=[model_spec],
            embedding_model="test-embed",
            result_file="results.jsonl",
        )
        return provider, "fake-model"

    monkeypatch.setattr(BenchmarkRunner, "initialize", fake_initialize)


def patch_glove(monkeypatch):
    monkeypatch.setattr(dat_module, "_get_glove_vectors", lambda config: DAT_VOCAB)


async def test_item_call_ids_are_logged(monkeypatch, tmp_path):
    provider = FakeProvider()
    patch_initialize(monkeypatch, tmp_path, provider)
    patch_glove(monkeypatch)

    result = await run_benchmarks(
        model_spec="fake:fake-model",
        suite="dat",
        conditions={"neutral": {"condition": "neutral", "num_samples": 2}},
        max_usd=1.0,
        config={"conditions": {}},
        runs_dir=str(tmp_path),
    )

    run_id = result["run_id"]
    calls = load_calls(run_id, str(tmp_path))
    logged_ids = {call.call_id for call in calls}
    items = result["results"]["dat"]["neutral"]["raw"].items

    assert items
    assert all(item["call_id"] in logged_ids for item in items)
    assert set(result["results"]["dat"]["neutral"]["raw"].call_ids) <= logged_ids


async def test_budget_exceeded_raises_and_manifest_accounts_cost(monkeypatch, tmp_path):
    provider = FakeProvider(cost=0.01)
    patch_initialize(monkeypatch, tmp_path, provider)
    patch_glove(monkeypatch)

    with pytest.raises(BudgetExceededError):
        await run_benchmarks(
            model_spec="fake:fake-model",
            suite="dat",
            conditions={"neutral": {"condition": "neutral", "num_samples": 2}},
            max_usd=0.001,
            config={"conditions": {}},
            runs_dir=str(tmp_path),
        )

    calls = load_calls("test-run", str(tmp_path))
    manifest = load_manifest("test-run", str(tmp_path))

    assert manifest is not None
    assert manifest.status == "failed"
    assert calls
    assert manifest.total_cost_usd >= sum(call.cost_usd for call in calls)


async def test_results_file_has_one_line_per_benchmark_condition(monkeypatch, tmp_path):
    provider = FakeProvider()
    patch_initialize(monkeypatch, tmp_path, provider)
    patch_glove(monkeypatch)

    conditions = {
        "neutral": {"condition": "neutral", "num_samples": 1},
        "creative": {"condition": "creative", "num_samples": 1},
    }
    await run_benchmarks(
        model_spec="fake:fake-model",
        suite="dat",
        conditions=conditions,
        max_usd=1.0,
        config={"conditions": {}},
        runs_dir=str(tmp_path),
    )

    results_file = tmp_path / "test-run" / "results.jsonl"
    rows = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines() if line]

    assert len(rows) == len(conditions)
    for row in rows:
        assert set(row) == {"benchmark", "condition", "model", "raw", "scores"}
        assert set(row["raw"]) == {"benchmark_id", "model", "items", "call_ids", "metadata"}
        assert set(row["scores"]) == {
            "benchmark_id",
            "model",
            "metrics",
            "confidence_intervals",
            "metadata",
        }


async def test_provider_exception_is_logged_and_run_fails(monkeypatch, tmp_path):
    provider = FakeProvider(error="boom")
    patch_initialize(monkeypatch, tmp_path, provider)

    class ExplodingBenchmark:
        id = "dat"
        version = "test"

        def prompts(self):
            return [BenchmarkItem(id="explode", prompt="explode")]

        async def run(self, provider, model, cfg):
            await provider.complete(LLMRequest(model=model, messages=[{"role": "user", "content": "hi"}]))

        def score(self, raw):
            raise AssertionError("score should not run")

    monkeypatch.setattr("cbf.runner.get_benchmark", lambda name: ExplodingBenchmark())

    with pytest.raises(RuntimeError, match="boom"):
        await run_benchmarks(
            model_spec="fake:fake-model",
            suite="dat",
            conditions={"neutral": {"condition": "neutral", "num_samples": 1}},
            max_usd=1.0,
            config={"conditions": {}},
            runs_dir=str(tmp_path),
        )

    calls = load_calls("test-run", str(tmp_path))
    manifest = load_manifest("test-run", str(tmp_path))

    assert len(calls) == 1
    assert calls[0].error == "boom"
    assert manifest is not None
    assert manifest.status == "failed"
    assert manifest.error == "boom"


def test_dat_invalid_sample_scores_zero_and_compliance(monkeypatch):
    bench = DATBenchmark()

    def fake_vectors(words):
        if words[0] == "apple":
            return np.array([[1.0, 0.0]])
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(bench, "_get_word_vectors", fake_vectors)

    raw = RawResults(
        benchmark_id="dat",
        model="fake-model",
        items=[
            {"sample_idx": 0, "valid": True, "words": ["apple", "banana"]},
            {"sample_idx": 1, "valid": True, "words": ["cat", "dog"]},
        ],
        call_ids=[],
        metadata={"condition": "neutral"},
    )

    scores = bench.score(raw)

    assert scores.metrics["valid_samples"] == 1
    assert scores.metrics["total_samples"] == 2
    assert scores.metrics["compliance_rate"] == 0.5
    assert scores.metrics["oov_terms"] == 1
    assert scores.metrics["dat_score"] == 50.0
    assert "dat_score" in scores.confidence_intervals
