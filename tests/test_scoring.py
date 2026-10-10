import sys

sys.path.insert(0, r'D:\p\eureka\src')

from unittest import mock

import numpy as np

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


def test_aut_score():
    from cbf.benchmarks.aut import AUTBenchmark, AUTConfig

    bench = AUTBenchmark(AUTConfig())
    items = [
        {"object": "brick", "sample_idx": 0, "valid": True, "uses": ["as a paperweight", "as a doorstop", "as a hammer"]},
        {"object": "brick", "sample_idx": 1, "valid": True, "uses": ["as a boat anchor", "to build a small oven", "as a musical instrument"]},
    ]
    raw = make_raw("aut", items)
    rng = np.random.default_rng(2)
    dim = 300
    pool = {}

    def fake_vectors(words):
        out = []
        for w in words:
            if w not in pool:
                pool[w] = rng.dirichlet(np.ones(dim)).astype(np.float32)
            out.append(pool[w])
        return np.array(out)

    with mock.patch.object(bench, "_get_use_vectors", side_effect=fake_vectors):
        scores = bench.score(raw)
    assert scores.metrics["valid_samples"] == 2
    assert scores.metrics["fluency"] > 0
    assert "originality" in scores.confidence_intervals


def test_diversity_score():
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


def test_diversity_score_empty():
    from cbf.benchmarks.diversity import DiversityBenchmark

    bench = DiversityBenchmark()
    scores = bench.score(make_raw("diversity", [{"error": "boom"}]))
    assert scores.metrics["diversity_score"] == 0.0


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
