import sys

import pytest

sys.path.insert(0, r'D:\p\eureka\src')

from cbf.benchmarks import get_benchmark, list_benchmarks
from cbf.benchmarks.aut import AUTBenchmark
from cbf.benchmarks.dat import DATBenchmark
from cbf.benchmarks.diversity import DiversityBenchmark


def test_dat_prompts():
    bench = DATBenchmark()
    prompts = bench.prompts()
    assert len(prompts) == 2
    assert prompts[0].id == "dat_neutral"
    assert prompts[1].id == "dat_creative"
    assert "neutral" in prompts[0].metadata["condition"]
    assert "creative" in prompts[1].metadata["condition"]


def test_dat_parse_words():
    bench = DATBenchmark()
    words = bench._parse_words("apple, banana, cherry, date, elderberry, fig, grape, honeydew, kiwi, lemon")
    assert len(words) == 10
    assert words == ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]

    words = bench._parse_words("  cat , dog ,  bird  ")
    assert words == ["cat", "dog", "bird"]


def test_aut_prompts():
    bench = AUTBenchmark()
    prompts = bench.prompts()
    assert len(prompts) == 20  # 10 objects * 2 conditions
    assert all(p.metadata["object"] in bench.config.objects for p in prompts)


def test_aut_parse_uses():
    bench = AUTBenchmark()
    text = "1. Use as a paperweight\n2. Use as a doorstop\n3. Build a small wall"
    uses = bench._parse_uses(text)
    assert len(uses) == 3
    assert uses[0] == "use as a paperweight"


def test_diversity_prompts():
    bench = DiversityBenchmark()
    prompts = bench.prompts()
    assert len(prompts) == 5
    assert all(p.id.startswith("diversity_") for p in prompts)


def test_distinct_n():
    bench = DiversityBenchmark()
    texts = ["the cat sat on the mat", "the dog ran in the park"]
    d1 = bench._compute_distinct_n(texts, 1)
    d2 = bench._compute_distinct_n(texts, 2)
    assert 0 <= d1 <= 1
    assert 0 <= d2 <= 1


def test_self_bleu():
    bench = DiversityBenchmark()
    texts = ["the cat sat on the mat", "the cat sat on the mat"]
    score = bench._compute_self_bleu(texts)
    assert abs(score - 1.0) < 1e-10  # identical texts

    texts = ["the cat sat on the mat", "the dog ran in the park"]
    score = bench._compute_self_bleu(texts)
    assert score < 1.0


def test_clustering():
    bench = DiversityBenchmark()
    import numpy as np
    emb = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
    ])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    clusters = bench._cluster_by_similarity(emb, 0.9)
    assert len(clusters) == 2


def test_registry():
    assert "dat" in list_benchmarks()
    assert "aut" in list_benchmarks()
    assert "diversity" in list_benchmarks()

    dat = get_benchmark("dat")
    assert isinstance(dat, DATBenchmark)

    with pytest.raises(ValueError):
        get_benchmark("unknown")


if __name__ == "__main__":
    test_dat_prompts()
    test_dat_parse_words()
    test_aut_prompts()
    test_aut_parse_uses()
    test_diversity_prompts()
    test_distinct_n()
    test_self_bleu()
    test_clustering()
    test_registry()
    print("All benchmark tests passed!")
