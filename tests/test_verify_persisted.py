import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import cbf.benchmarks as benchmarks
from cbf.benchmarks.base import RawResults, Scores
from cbf.runner.verify import verify_run

CONDITION_LENGTHS = {
    "neutral": [2, 4],
    "creative": [10, 20],
}


class FakeBenchmark:
    id = "fake"
    version = "1.0.0"

    def prompts(self):
        return []

    async def run(self, provider, model: str, cfg: dict):
        raise NotImplementedError

    def score(self, raw: RawResults) -> Scores:
        lengths = [
            len(item.get("response", ""))
            for item in raw.items
            if not item.get("error")
        ]
        mean_length = sum(lengths) / len(lengths) if lengths else 0.0
        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "mean_length": mean_length,
                "n_items": float(len(raw.items)),
            },
            confidence_intervals={"mean_length": (mean_length, mean_length)},
            metadata={"condition": raw.metadata.get("condition", "unknown")},
        )


class ExplodingBenchmark(FakeBenchmark):
    id = "fake"

    def score(self, raw: RawResults) -> Scores:
        raise RuntimeError("embeddings unavailable offline")


def _fake_factory(config=None):
    return FakeBenchmark()


def _exploding_factory(config=None):
    return ExplodingBenchmark()


def _build_run(
    tmp_path,
    run_id="run1",
    tamper=None,
    write_results=True,
    benchmark="fake",
    factory=None,
):
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scorer = (factory or _fake_factory)()
    calls = []
    entries = []

    for condition, lengths in CONDITION_LENGTHS.items():
        items = []
        call_ids = []
        for index, length in enumerate(lengths):
            call_id = f"{condition}-{index}"
            call_ids.append(call_id)
            response_text = "x" * length
            calls.append({
                "run_id": run_id,
                "call_id": call_id,
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "provider": "fake",
                "model": "fake-model",
                "request": {"prompt": condition},
                "response": {"text": response_text},
                "latency_s": 0.1,
                "cost_usd": 0.01,
                "input_tokens": 1,
                "output_tokens": length,
                "error": None,
            })
            items.append({
                "call_id": call_id,
                "response": response_text,
                "latency_s": 0.1,
                "tokens": {"input": 1, "output": length},
            })

        raw = {
            "benchmark_id": benchmark,
            "model": "fake-model",
            "items": items,
            "call_ids": call_ids,
            "metadata": {"condition": condition},
        }
        scores = scorer.score(RawResults(**raw))
        stored_scores = {
            "benchmark_id": scores.benchmark_id,
            "model": scores.model,
            "metrics": dict(scores.metrics),
            "confidence_intervals": {
                metric: list(interval)
                for metric, interval in scores.confidence_intervals.items()
            },
            "metadata": dict(scores.metadata),
        }
        if tamper and tamper[0] == condition:
            stored_scores["metrics"][tamper[1]] = tamper[2]

        entries.append({
            "benchmark": benchmark,
            "condition": condition,
            "model": "fake-model",
            "raw": raw,
            "scores": stored_scores,
        })

    with open(run_dir / "calls.jsonl", "w", encoding="utf-8") as f:
        for call in calls:
            f.write(json.dumps(call) + "\n")

    if write_results:
        with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    manifest = {
        "run_id": run_id,
        "started_utc": "2026-01-01T00:00:00+00:00",
        "ended_utc": "2026-01-01T00:10:00+00:00",
        "models": ["fake-model"],
        "provider_versions": {},
        "config_hash": "",
        "prompt_set_hashes": {},
        "embedding_model": "",
        "judge_models": [],
        "conditions": {},
        "total_cost_usd": sum(call["cost_usd"] for call in calls),
        "total_calls": len(calls),
        "status": "completed",
        "suite": "dat",
        "error": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return run_id


def test_verify_persisted_run_succeeds(tmp_path, monkeypatch):
    monkeypatch.setitem(benchmarks.BENCHMARK_REGISTRY, "fake", _fake_factory)
    run_id = _build_run(tmp_path)

    result = verify_run(run_id, str(tmp_path))

    assert result["success"] is True
    assert result["verified_results"] == 2
    assert result["mismatches"] == []
    assert result["missing_call_ids"] == []
    assert result["call_count_match"] is True
    assert result["cost_match"] is True
    assert "error" not in result


def test_verify_detects_tampered_metric(tmp_path, monkeypatch):
    monkeypatch.setitem(benchmarks.BENCHMARK_REGISTRY, "fake", _fake_factory)
    run_id = _build_run(tmp_path, tamper=("neutral", "mean_length", 999.0))

    result = verify_run(run_id, str(tmp_path))

    assert result["success"] is False
    assert len(result["mismatches"]) == 1
    mismatch = result["mismatches"][0]
    assert mismatch["condition"] == "neutral"
    assert mismatch["metric"] == "mean_length"
    assert mismatch["stored"] == 999.0
    assert mismatch["computed"] == pytest.approx(3.0)
    assert mismatch["reason"] == "value_mismatch"
    assert "error" in result


def test_verify_fails_without_results_file(tmp_path, monkeypatch):
    monkeypatch.setitem(benchmarks.BENCHMARK_REGISTRY, "fake", _fake_factory)
    run_id = _build_run(tmp_path, write_results=False)

    result = verify_run(run_id, str(tmp_path))

    assert result["success"] is False
    assert "no persisted results (results.jsonl)" in result["error"]


def test_verify_reports_recompute_error(tmp_path, monkeypatch):
    monkeypatch.setitem(benchmarks.BENCHMARK_REGISTRY, "fake", _exploding_factory)
    run_id = _build_run(tmp_path)

    result = verify_run(run_id, str(tmp_path))

    assert result["success"] is False
    assert result["verified_results"] == 0
    assert len(result["recompute_errors"]) == 2
    assert "embeddings unavailable offline" in result["recompute_errors"][0]["recompute_error"]
    assert "error" in result
