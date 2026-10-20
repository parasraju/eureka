import asyncio
import sys

sys.path.insert(0, r'D:\p\eureka\src')

from cbf.judging.panel import (
    PairwiseResult,
    compute_bradley_terry,
    compute_elo,
    compute_inter_judge_agreement,
    compute_model_matches,
    compute_model_ratings,
    parse_winner,
    resolve_pairs,
)
from cbf.providers.base import LLMRequest
from cbf.providers.rate_limiter import (
    RateLimiter,
    rate_limited_complete,
)


def test_parse_winner_standalone():
    assert parse_winner("A") == "a"
    assert parse_winner("B") == "b"
    assert parse_winner("TIE") == "tie"
    assert parse_winner("**A**\nBecause it is vivid.") == "a"
    assert parse_winner("b: more original imagery") == "b"
    assert parse_winner('"A."') == "a"
    assert parse_winner("(B)") == "b"
    assert parse_winner("**TIE**") == "tie"
    assert parse_winner("A because the imagery is stronger") == "a"


def test_parse_winner_is_strict_about_leading_token():
    assert parse_winner("Text A: more vivid imagery") == "invalid"
    assert parse_winner("The answer is A. It flows better.") == "invalid"
    assert parse_winner("Answer: B") == "invalid"
    assert parse_winner("I cannot decide.") == "invalid"
    assert parse_winner("") == "invalid"
    assert parse_winner("A/B") == "invalid"


def test_resolve_pairs_ignores_invalid():
    rs = [
        PairwiseResult("a", "b", "invalid", "j1", False),
        PairwiseResult("a", "b", "a", "j1", True),
    ]
    resolved = resolve_pairs(rs)
    assert len(resolved) == 1
    assert resolved[0].winner == "a"

    only_invalid = resolve_pairs([PairwiseResult("a", "b", "invalid", "j1", False)])
    assert only_invalid[0].winner == "invalid"


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
    results = [
        PairwiseResult("a", "b", "tie", "judge1", False),
    ]
    ratings = compute_elo(results)
    assert ratings == {"a": 1000.0, "b": 1000.0}
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("b", "c", "a", "judge1", False),
        PairwiseResult("a", "c", "tie", "judge1", False),
    ]
    ratings = compute_elo(results)
    assert ratings["a"] > 1000
    assert ratings["c"] < 1000


def test_elo_ignores_invalid_judgment():
    results = [
        PairwiseResult("a", "b", "invalid", "judge1", False),
    ]
    assert compute_elo(results) == {"a": 1000.0, "b": 1000.0}


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
    assert agreement["invalid_rate"] == 0.0


def test_agreement_reports_invalid_rate():
    results = [
        PairwiseResult("a", "b", "a", "judge1", False),
        PairwiseResult("a", "b", "invalid", "judge2", False),
    ]
    agreement = compute_inter_judge_agreement(results)
    assert agreement["invalid_rate"] == 0.5


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


def test_compute_model_matches_cross_model_only():
    results = [
        PairwiseResult("m1::0", "m2::0", "a", "j1", False),
        PairwiseResult("m1::0", "m2::1", "b", "j1", False),
        PairwiseResult("m1::1", "m2::1", "tie", "j1", False),
        PairwiseResult("m1::0", "m1::1", "a", "j1", False),
    ]
    matches = compute_model_matches(results)
    assert matches["m1"] == {"wins": 1, "losses": 1, "ties": 1}
    assert matches["m2"] == {"wins": 1, "losses": 1, "ties": 1}


def test_compute_model_ratings_cross_model_deterministic():
    results = [
        PairwiseResult("m1::0", "m2::0", "a", "j1", False),
        PairwiseResult("m1::0", "m2::1", "a", "j1", False),
        PairwiseResult("m1::1", "m2::0", "a", "j1", False),
        PairwiseResult("m1::1", "m2::1", "tie", "j1", False),
        PairwiseResult("m1::0", "m1::1", "a", "j1", False),
    ]
    model_of_item = {
        "m1::0": "m1", "m1::1": "m1",
        "m2::0": "m2", "m2::1": "m2",
    }
    ratings = compute_model_ratings(results, model_of_item)
    assert set(ratings) == {"elo", "bt", "win_rate", "invalid_rate"}
    assert ratings["elo"]["m1"] > ratings["elo"]["m2"]
    assert ratings["bt"]["m1"] > ratings["bt"]["m2"]
    assert ratings["win_rate"]["m1"] == (3 + 0.5) / 4
    assert ratings["win_rate"]["m2"] == (0 + 0.5) / 4
    assert ratings["invalid_rate"] == 0.0
    assert compute_model_ratings(results, model_of_item) == ratings


class SlowProvider:
    def __init__(self, delay=0.2):
        self.delay = delay
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        else:
            await self.release.wait()
        self.active -= 1
        return "ok"


def test_rate_limiter_enforces_concurrency_one():
    async def scenario():
        limiter = RateLimiter(max_concurrency=1)
        provider = SlowProvider(delay=0.01)
        req = LLMRequest(model="m", messages=[])
        await asyncio.gather(*[
            rate_limited_complete(provider, req, limiter) for _ in range(5)
        ])
        return provider

    provider = asyncio.run(scenario())
    assert provider.calls == 5
    assert provider.peak == 1


def test_rate_limiter_timeout_raises_without_hanging():
    async def scenario():
        limiter = RateLimiter(max_concurrency=1, timeout=0.05)
        provider = SlowProvider(delay=0)
        req = LLMRequest(model="m", messages=[])
        first = asyncio.create_task(rate_limited_complete(provider, req, limiter))
        await provider.started.wait()
        timed_out = False
        try:
            await rate_limited_complete(provider, req, limiter)
        except TimeoutError:
            timed_out = True
        provider.release.set()
        await first
        return timed_out

    assert asyncio.run(scenario()) is True


if __name__ == "__main__":
    test_parse_winner_standalone()
    test_parse_winner_is_strict_about_leading_token()
    test_resolve_pairs_ignores_invalid()
    test_resolve_pairs_agreement_wins()
    test_resolve_pairs_disagreement_is_tie()
    test_resolve_pairs_single_decided_among_ties()
    test_resolve_pairs_keeps_distinct_judges()
    test_elo_penalizes_both_on_tie()
    test_elo_ignores_invalid_judgment()
    test_bt_counts_ties_as_half_wins()
    test_agreement_low_with_single_judge()
    test_agreement_reports_invalid_rate()
    test_compute_elo()
    test_compute_elo_ties()
    test_compute_bradley_terry()
    test_inter_judge_agreement()
    test_inter_judge_agreement_disagreement()
    test_compute_model_matches_cross_model_only()
    test_compute_model_ratings_cross_model_deterministic()
    test_rate_limiter_enforces_concurrency_one()
    test_rate_limiter_timeout_raises_without_hanging()
    print("All judging tests passed!")
