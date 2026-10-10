import sys

sys.path.insert(0, r'D:\p\eureka\src')

import numpy as np

from cbf.metrics import (
    bootstrap_ci,
    bootstrap_comparison,
    cluster_by_similarity,
    composite,
    distinct_n,
    human_percentile,
    mean_ci,
    mean_pairwise_distance,
    normalize,
    self_bleu,
    t_ci,
)


def test_bootstrap_ci_deterministic():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert bootstrap_ci(data, seed=1) == bootstrap_ci(data, seed=1)
    lo, hi = bootstrap_ci(data, seed=1)
    assert lo <= hi
    assert 0 < lo < hi < 10


def test_bootstrap_ci_singleton():
    assert bootstrap_ci([5.0]) == (5.0, 5.0)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_t_ci():
    data = [1.0, 2.0, 3.0, 4.0]
    lo, hi = t_ci(data)
    assert lo <= hi


def test_mean_ci_methods():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = mean_ci(data, method="bootstrap")
    t = mean_ci(data, method="t")
    assert b[0] <= b[1]
    assert t[0] <= t[1]


def test_bootstrap_comparison():
    strong = [10.0] * 20
    weak = [1.0] * 20
    res = bootstrap_comparison(strong, weak, seed=0)
    assert res["diff"] > 8.0
    assert res["significant"] is True


def test_distinct_n():
    texts = ["the cat sat on the mat", "the dog ran in the park"]
    assert 0 < distinct_n(texts, 1) < 1
    assert 0 < distinct_n(texts, 2) <= 1
    assert distinct_n([], 1) == 0.0


def test_self_bleu_identical_is_one():
    texts = ["the cat sat on the mat", "the cat sat on the mat"]
    assert abs(self_bleu(texts) - 1.0) < 1e-10
    assert self_bleu(["only one"]) == 0.0


def test_mean_pairwise_distance():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    d = mean_pairwise_distance(emb)
    # pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=1.0 -> mean 2/3
    assert abs(d - 2.0 / 3.0) < 1e-6
    assert mean_pairwise_distance(np.array([[1.0, 0.0]])) == 0.0


def test_cluster_by_similarity():
    emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    clusters = cluster_by_similarity(emb, 0.9)
    assert len(clusters) == 2


def test_normalize():
    assert normalize(50.0, 0.0, 100.0) == 50.0
    assert normalize(150.0, 0.0, 100.0) == 100.0
    assert normalize(5.0, 10.0, 10.0) == 0.0


def test_human_percentile():
    humans = [20.0, 30.0, 40.0, 50.0]
    assert abs(human_percentile(30.0, humans) - 50.0) < 1e-9
    assert human_percentile(100.0, humans) == 100.0
    assert human_percentile(30.0, []) == 0.0


def test_composite():
    assert abs(composite({"a": 1.0, "b": 3.0}) - 2.0) < 1e-9
    assert abs(composite({"a": 1.0, "b": 3.0}, {"a": 3.0, "b": 1.0}) - 1.5) < 1e-9
    assert composite({}) == 0.0
