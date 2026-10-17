import sys

sys.path.insert(0, r'D:\p\eureka\src')

from cbf.judging.panel import (
    PairwiseResult,
    compute_bradley_terry,
    compute_elo,
    compute_inter_judge_agreement,
    parse_winner,
    resolve_pairs,
)


def test_parse_winner_standalone():
    assert parse_winner("A") == "a"
    assert parse_winner("B") == "b"
    assert parse_winner("TIE") == "tie"
    assert parse_winner("**A**\nBecause it is vivid.") == "a"
    assert parse_winner("b: more original imagery") == "b"


def test_parse_winner_prose_and_undecidable():
    assert parse_winner("The answer is A. It flows better.") == "a"
    # Undecidable / empty responses must not be silently treated as a win.
    assert parse_winner("") == "tie"
    assert parse_winner("I cannot decide.") == "tie"


def test_resolve_pairs_agreement_wins():
    rs = [
        PairwiseResult("a", "b", "a", "j1", False),
        PairwiseResult("a", "b", "a", "j1", True),
    ]
    resolved = resolve_pairs(rs)
    assert len(resolved) == 1
    assert resolved[0].winner == "a"


def test_resolve_pairs_disagreement_is_tie():
    rs = [
        PairwiseResult("a", "b", "a", "j1", False),
        PairwiseResult("a", "b", "b", "j1", True),
    ]
    resolved = resolve_pairs(rs)
    assert len(resolved) == 1
    assert resolved[0].winner == "tie"


def test_resolve_pairs_single_decided_among_ties():
    rs = [
        PairwiseResult("a", "b", "tie", "j1", False),
        PairwiseResult("a", "b", "a", "j1", True),
    ]
    resolved = resolve_pairs(rs)
    assert len(resolved) == 1
    assert resolved[0].winner == "a"


def test_resolve_pairs_keeps_distinct_judges():
    rs = [
        PairwiseResult("a", "b", "a", "j1", False),
        PairwiseResult("a", "b", "b", "j2", False),
    ]
    resolved = resolve_pairs(rs)
    assert len(resolved) == 2


def test_elo_penalizes_both_on_tie():
    # A clear tie should pull both items toward the mean, not be ignored.
    results = [
        PairwiseResult("a", "b", "tie", "judge1", False),
    ]
    ratings = compute_elo(results)
    assert ratings == {"a": 1000.0, "b": 1000.0}
    # A tie between a strong and weak item moves them together.
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("b", "c", "a", "judge1", False),
        PairwiseResult("a", "c", "tie", "judge1", False),
    ]
    ratings = compute_elo(results)
    assert ratings["a"] > 1000
    assert ratings["c"] < 1000


def test_bt_counts_ties_as_half_wins():
    results = [
        PairwiseResult("a", "b", "tie", "judge1", False),
        PairwiseResult("b", "c", "a", "judge1", False),
    ]
    ratings = compute_bradley_terry(results)
    assert ratings["b"] > ratings["c"]
    assert ratings["b"] >= ratings["a"]


def test_agreement_low_with_single_judge():
    results = [PairwiseResult("a", "b", "a", "judge1", False)]
    agreement = compute_inter_judge_agreement(results)
    assert agreement["num_judges"] == 1
    assert agreement["krippendorff_alpha"] == 0.0
    assert agreement["mean_spearman"] == 0.0


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
    test_parse_winner_standalone()
    test_parse_winner_prose_and_undecidable()
    test_resolve_pairs_agreement_wins()
    test_resolve_pairs_disagreement_is_tie()
    test_resolve_pairs_single_decided_among_ties()
    test_resolve_pairs_keeps_distinct_judges()
    test_elo_penalizes_both_on_tie()
    test_bt_counts_ties_as_half_wins()
    test_agreement_low_with_single_judge()
    test_compute_elo()
    test_compute_elo_ties()
    test_compute_bradley_terry()
    test_inter_judge_agreement()
    test_inter_judge_agreement_disagreement()
    print("All judging tests passed!")
