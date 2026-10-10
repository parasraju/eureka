import sys

sys.path.insert(0, r'D:\p\eureka\src')

from cbf.judging.bias import (
    blind_texts,
    length_bias_correlation,
    length_controlled_ratings,
    position_win_rate,
    swap_consistency,
)
from cbf.judging.calibration import (
    apply_severity_correction,
    calibrated_panel_scores,
    estimate_ability_and_severity,
    judge_severities,
)
from cbf.judging.panel import PairwiseResult


def result(a, b, winner, judge, swapped=False):
    return PairwiseResult(item_a_id=a, item_b_id=b, winner=winner, judge=judge, position_swapped=swapped)


def test_position_win_rate_balanced():
    rs = [
        result("x", "y", "x", "j1", False),
        result("x", "y", "y", "j1", True),
        result("x", "y", "x", "j1", False),
        result("x", "y", "y", "j1", True),
    ]
    r = position_win_rate(rs)
    assert abs(r["overall"] - 0.5) < 1e-9


def test_position_win_rate_first_aligned():
    # No swaps: first-listed always wins -> strong position artifact detected
    rs = [result("a", "b", "a", "j1", False), result("a", "b", "a", "j1", False)]
    r = position_win_rate(rs)
    assert r["overall"] == 1.0
    assert r["n_decided"] == 2


def test_swap_consistency():
    rs = [
        result("a", "b", "a", "j1", False),
        result("a", "b", "a", "j1", True),
        result("b", "c", "a", "j1", False),
        result("b", "c", "b", "j1", True),
    ]
    r = swap_consistency(rs)
    assert r["consistency"] == 0.5
    assert r["pairs_compared"] == 2


def test_length_bias_correlation():
    lengths = {"a": 100, "b": 200, "c": 50, "d": 400, "e": 300}
    ratings = {"a": 5.0, "b": 6.0, "c": 4.0, "d": 9.0, "e": 8.0}
    r = length_bias_correlation(lengths, ratings)
    assert r["n"] == 5
    assert 0.0 < r["spearman"] <= 1.0
    assert length_bias_correlation({"a": 10}, {"a": 5.0})["n"] < 3


def test_length_controlled_ratings_preserves_spread():
    lengths = {"a": 100, "b": 200, "c": 50, "d": 400, "e": 300}
    ratings = {"a": 5.0, "b": 6.0, "c": 4.0, "d": 9.0, "e": 8.0}
    controlled = length_controlled_ratings(ratings, lengths)
    assert set(controlled) == set(ratings)
    # mean preserved
    assert abs(sum(controlled.values()) / len(controlled) - sum(ratings.values()) / len(ratings)) < 1e-9
    # length advantage removed: 'd' (longest) no longer necessarily top-best vs 'a'
    diff_raw = ratings["d"] - ratings["a"]
    diff_ctrl = controlled["d"] - controlled["a"]
    assert diff_ctrl < diff_raw


def test_blind_texts():
    texts = {"model_a": "hello", "model_b": "world", "model_c": "again"}
    labels, relabeled = blind_texts(texts, seed=42)
    assert set(labels) == set(texts)
    assert set(relabeled.keys()) == {chr(ord("A") + i) for i in range(len(texts))}
    assert set(labels.values()) == set(relabeled.keys())
    assert set(relabeled.values()) == set(texts.values())


def test_judge_severities_harsh_vs_lenient():
    # judge1 always picks item_a, judge2 always picks item_b
    rs = []
    for _ in range(5):
        rs.append(result("good", "bad", "a", "judge_lenient", False))
        rs.append(result("good", "bad", "a", "judge_lenient", True))
        rs.append(result("good", "bad", "b", "judge_harsh", False))
        rs.append(result("good", "bad", "b", "judge_harsh", True))
    sev = judge_severities(rs)
    assert set(sev) == {"judge_lenient", "judge_harsh"}


def test_apply_severity_correction():
    ratings = {
        "j1": {"a": 1100.0, "b": 1050.0},
        "j2": {"a": 980.0, "b": 950.0},
    }
    severities = {"j1": 50.0, "j2": -50.0}
    corrected = apply_severity_correction(ratings, severities)
    # After correction both judges agree on ordering and scores line up
    assert corrected["a"] > corrected["b"]
    assert abs(corrected["a"] - (1100 - 50 + 980 + 50) / 2) < 1e-9


def test_estimate_ability_and_severity():
    rs = [result("s", "w", "a", "j", False) for _ in range(10)]
    out = estimate_ability_and_severity(rs)
    assert out["ability"]["s"] > out["ability"]["w"]
    assert out["log_loss"] >= 0.0
    assert set(out["severity"]) == {"j"}


def test_calibrated_panel_scores():
    rs = [result("a", "b", "a", "j1", False) for _ in range(3)] + [result("a", "b", "a", "j2", True) for _ in range(3)]
    out = calibrated_panel_scores(rs)
    assert out["scores"]["a"] > out["scores"]["b"]
    assert out["reliable"] is True


def test_judge_severities_empty():
    assert judge_severities([]) == {}
