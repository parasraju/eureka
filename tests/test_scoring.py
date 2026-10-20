from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from cbf.benchmarks.base import RawResults


def make_raw(bench_id, items, model="test-model"):
    return RawResults(benchmark_id=bench_id, model=model, items=items, call_ids=[], metadata={})


def test_dat_score_smoke():
    from cbf.benchmarks.dat import DATBenchmark

    bench = DATBenchmark()
    items = [
        {"sample_idx": i, "response": "", "valid": True, "words": ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "kiwi", "lemon", "mango"]}
        for i in range(5)
    ]
    raw = make_raw("dat", items)

    def fake_vectors(words):
        rng = np.random.default_rng(1)
        return rng.dirichlet(np.ones(300), size=len(words)).astype(np.float32)

    with mock.patch.object(bench, "_get_word_vectors", side_effect=fake_vectors):
        scores = bench.score(raw)
    assert scores.metrics["valid_samples"] == 5
    assert scores.metrics["dat_score"] > 0
    assert "dat_score" in scores.confidence_intervals


def test_dat_score_no_valid():
    from cbf.benchmarks.dat import DATBenchmark

    bench = DATBenchmark()
    scores = bench.score(make_raw("dat", [{"sample_idx": 0, "valid": False, "words": []}]))
    assert scores.metrics["valid_samples"] == 0


def _aut_items():
    return [
        {"object": "brick", "sample_idx": 0, "valid": True, "uses": ["as a paperweight", "as a doorstop", "as a hammer"]},
        {"object": "brick", "sample_idx": 1, "valid": True, "uses": ["as a boat anchor", "to build a small oven", "as a musical instrument"]},
    ]


def test_aut_multiword_uses_use_embedding_seam():
    from cbf.benchmarks.aut import AUTBenchmark, AUTConfig

    bench = AUTBenchmark(AUTConfig())
    raw = make_raw("aut", _aut_items())
    rng = np.random.default_rng(2)
    pool: dict[str, np.ndarray] = {}
    seen_texts: list[str] = []
    seen_models: list[str] = []

    def fake_embed(texts, model_name):
        nonlocal pool
        seen_texts.extend(texts)
        seen_models.append(model_name)
        out = []
        for text in texts:
            if text not in pool:
                pool[text] = rng.dirichlet(np.ones(32))
            out.append(pool[text])
        return np.array(out)

    with mock.patch("cbf.metrics.embed_texts", side_effect=fake_embed):
        scores = bench.score(raw)

    assert "as a paperweight" in seen_texts
    assert all(name == bench.config.st_model for name in seen_models)
    assert np.isfinite(scores.metrics["originality"])
    assert scores.metrics["originality"] >= 0.0
    assert scores.metrics["elaboration"] > 0.0
    assert "originality" in scores.confidence_intervals


def test_aut_score():
    from cbf.benchmarks.aut import AUTBenchmark, AUTConfig

    bench = AUTBenchmark(AUTConfig())
    raw = make_raw("aut", _aut_items())
    rng = np.random.default_rng(2)
    dim = 32
    pool: dict[str, np.ndarray] = {}

    def fake_vectors(words):
        out = []
        for word in words:
            if word not in pool:
                pool[word] = rng.dirichlet(np.ones(dim)).astype(np.float32)
            out.append(pool[word])
        return np.array(out)

    with mock.patch.object(bench, "_get_use_vectors", side_effect=fake_vectors):
        scores = bench.score(raw)
    assert scores.metrics["valid_samples"] == 2
    assert scores.metrics["fluency"] > 0
    assert np.isfinite(scores.metrics["originality"])
    assert 1 <= scores.metrics["flexibility"] <= 3
    assert "originality" in scores.confidence_intervals


def test_aut_embedding_failure_raises():
    from cbf.benchmarks.aut import AUTBenchmark, AUTConfig

    bench = AUTBenchmark(AUTConfig())
    raw = make_raw("aut", _aut_items())

    with mock.patch("cbf.metrics.embed_texts", side_effect=RuntimeError("no model")):
        with pytest.raises(RuntimeError, match="AUT embeddings failed"):
            bench.score(raw)


class _FakeJudgeProvider:
    def __init__(self, text: str = "7"):
        self.text = text
        self.requests = []

    async def complete(self, req):
        self.requests.append(req)
        return SimpleNamespace(
            text=self.text,
            request_id="judge-1",
            latency_s=0.0,
            input_tokens=0,
            output_tokens=0,
        )


def test_diversity_score_without_quality_omits_composite():
    from cbf.benchmarks.diversity import DiversityBenchmark

    bench = DiversityBenchmark()
    items = [
        {"prompt_idx": 0, "text": "the cat sat on the mat", "generation_idx": i,
         "latency_s": 0.1, "tokens": {}, "call_id": f"c{i}"}
        for i in range(4)
    ]
    raw = make_raw("diversity", items)

    def fake_embeddings(texts):
        rng = np.random.default_rng(3)
        return rng.normal(size=(len(texts), 128))

    with mock.patch.object(bench, "_get_embeddings", side_effect=fake_embeddings):
        scores = bench.score(raw)
    assert scores.metrics["diversity_score"] > 0
    assert scores.metrics["self_bleu"] >= 0
    assert scores.metrics["prompts_tested"] == 1
    assert scores.metrics["generations_per_prompt"] == 4.0
    assert "composite" not in scores.metrics
    assert "quality_score" not in scores.metrics
    assert scores.metadata["composite_available"] is False


async def test_diversity_composite_with_fake_judge():
    from cbf.benchmarks.diversity import DiversityBenchmark

    bench = DiversityBenchmark()
    items = [
        {"prompt_idx": 0, "text": f"generation {i}", "generation_idx": i,
         "latency_s": 0.1, "tokens": {}, "call_id": f"c{i}"}
        for i in range(4)
    ]
    raw = make_raw("diversity", items)

    def fake_embeddings(texts):
        rng = np.random.default_rng(4)
        return rng.normal(size=(len(texts), 64))

    provider = _FakeJudgeProvider("7")
    with mock.patch.object(bench, "_get_embeddings", side_effect=fake_embeddings):
        await bench.judge(raw, {"judges": ["fake:judge"]}, lambda spec: provider)
        scores = bench.score(raw)

    assert len(raw.metadata["quality"]) == 4
    assert scores.metrics["quality_score"] == pytest.approx(0.7)
    assert scores.metadata["composite_available"] is True
    assert scores.metrics["composite"] == pytest.approx(scores.metrics["diversity_score"] * 0.7)
    assert "composite" in scores.confidence_intervals


def test_diversity_private_prompt_set_is_seeded():
    from cbf.benchmarks.diversity import DiversityConfig

    config = DiversityConfig()
    first = config.private_prompt_set(7, 5)
    second = config.private_prompt_set(7, 5)
    other = config.private_prompt_set(8, 5)
    assert first == second
    assert first != other
    assert len(first) == 5
    assert len(set(first)) == 5
    assert first != config.prompts


async def test_diversity_run_uses_private_prompts():
    from cbf.benchmarks.diversity import DiversityBenchmark

    class FakeProvider:
        def __init__(self):
            self.prompts = []

        async def complete(self, req):
            self.prompts.append(req.messages[0]["content"])
            return SimpleNamespace(
                text="private output",
                request_id="gen-1",
                latency_s=0.0,
                input_tokens=0,
                output_tokens=0,
            )

    bench = DiversityBenchmark()
    provider = FakeProvider()
    raw = await bench.run(provider, "m", {"private_prompts": True, "seed": 3, "k_generations": 1})

    assert raw.metadata["prompt_provenance"] == "private-seeded"
    assert raw.metadata["seed"] == 3
    assert len(raw.items) == len(provider.prompts) == 5
    assert set(provider.prompts) == set(bench.config.private_prompt_set(3, 5))


def test_diversity_score_empty():
    from cbf.benchmarks.diversity import DiversityBenchmark

    bench = DiversityBenchmark()
    scores = bench.score(make_raw("diversity", [{"error": "boom"}]))
    assert scores.metrics["diversity_score"] == 0.0
    assert scores.metadata["composite_available"] is False


def test_writing_elo_score():
    from cbf.benchmarks.writing_elo import create_writing_elo_benchmark

    bench = create_writing_elo_benchmark()
    items = [
        {"prompt_idx": 0, "prompt_id": "writing_0", "text": "A robot learns to paint with feeling."},
        {"prompt_idx": 1, "prompt_id": "writing_1", "text": "Gravity sideways, stairs become waterfalls."},
    ]
    scores = bench.score(make_raw("writing_elo", items))
    assert scores.metrics["items_generated"] == 2
    assert scores.metrics["prompts_tested"] == 2


def test_writing_elo_score_empty():
    from cbf.benchmarks.writing_elo import create_writing_elo_benchmark

    bench = create_writing_elo_benchmark()
    scores = bench.score(make_raw("writing_elo", [{"error": "boom"}]))
    assert scores.metrics["elo_rating"] == 0.0
    assert scores.metrics["agreement"] == 0.0

def test_aut_invalid_samples_score_zero():
    from cbf.benchmarks.aut import AUTBenchmark, AUTConfig

    bench = AUTBenchmark(AUTConfig())
    raw = make_raw("aut", [
        {"object": "brick", "valid": True, "uses": ["a", "b", "c"]},
        {"object": "brick", "valid": False, "uses": []},
    ])

    with mock.patch.object(
        bench, "_get_use_vectors", side_effect=lambda ws: np.eye(len(ws), 300, dtype=np.float32)
    ):
        scores = bench.score(raw)

    assert scores.metrics["fluency"] == pytest.approx(1.5)
    assert scores.metrics["compliance_rate"] == pytest.approx(0.5)
    assert scores.metrics["valid_samples"] == 1


def test_diversity_penalizes_prompt_with_one_generation():
    from cbf.benchmarks.diversity import DiversityBenchmark

    bench = DiversityBenchmark()
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    def score(items):
        raw = make_raw("diversity", items)
        with mock.patch.object(bench, "_get_embeddings", side_effect=lambda ts: embeddings[: len(ts)]):
            return bench.score(raw).metrics

    one_prompt = score([
        {"prompt_idx": 0, "generation_idx": 0, "text": "aaaa"},
        {"prompt_idx": 0, "generation_idx": 1, "text": "bbbb"},
    ])
    mixed = score([
        {"prompt_idx": 0, "generation_idx": 0, "text": "aaaa"},
        {"prompt_idx": 0, "generation_idx": 1, "text": "bbbb"},
        {"prompt_idx": 1, "generation_idx": 0, "text": "cccc"},
    ])

    assert mixed["prompts_tested"] == 2
    assert mixed["diversity_score"] == pytest.approx(one_prompt["diversity_score"] / 2)
    assert mixed["generation_failure_rate"] == 0.0
