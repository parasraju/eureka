import sys
sys.path.insert(0, r'D:\p\eureka\src')

from cbf.judging.panel import (
    compute_elo,
    compute_bradley_terry,
    compute_inter_judge_agreement,
    PairwiseResult,
)


def test_compute_elo():
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("b", "c", "a", "judge1", False),
        PairwiseResult("a", "c", "a", "judge1", False),
    ]
    ratings = compute_elo(results)
    assert ratings["a"] > ratings["b"]
    assert ratings["b"] > ratings["c"]
    assert ratings["a"] > 1000
    assert ratings["c"] < 1000


def test_compute_elo_ties():
    results = [
        PairwiseResult("a", "b", "tie", "judge1", False),
        PairwiseResult("b", "c", "a", "judge1", False),
    ]
    ratings = compute_elo(results)
    assert ratings["b"] > ratings["c"]


def test_compute_bradley_terry():
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("b", "c", "a", "judge1", False),
    ]
    ratings = compute_bradley_terry(results)
    assert ratings["a"] > ratings["b"]
    assert ratings["b"] > ratings["c"]


def test_inter_judge_agreement():
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("a", "b", "a", "judge2", False),
        PairwiseResult("b", "c", "a", "judge1", False),
        PairwiseResult("b", "c", "a", "judge2", False),
        PairwiseResult("a", "c", "a", "judge1", False),
        PairwiseResult("a", "c", "a", "judge2", False),
    ]
    agreement = compute_inter_judge_agreement(results)
    assert agreement["krippendorff_alpha"] > 0.8
    assert agreement["mean_spearman"] > 0.8
    assert agreement["num_judges"] == 2
    assert agreement["num_items"] == 3


def test_inter_judge_agreement_disagreement():
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("a", "b", "b", "judge2", False),
        PairwiseResult("b", "c", "a", "judge1", False),
        PairwiseResult("b", "c", "b", "judge2", False),
    ]
    agreement = compute_inter_judge_agreement(results)
    assert agreement["krippendorff_alpha"] < 0.5
    assert agreement["mean_spearman"] < 0.5


if __name__ == "__main__":
    test_compute_elo()
    test_compute_elo_ties()
    test_compute_bradley_terry()
    test_inter_judge_agreement()
    test_inter_judge_agreement_disagreement()
    print("All judging tests passed!")