"""Bias controls for LLM-as-judge panels: position, length, and identity blindness."""

import hashlib
import random

import numpy as np
from scipy import stats

from cbf.judging.panel import PairwiseResult


def position_win_rate(results: list[PairwiseResult]) -> dict:
    """Win rate of whichever item was presented first (should be ~0.5 if unbiased)."""
    decided = [r for r in results if r.winner != "tie"]
    if not decided:
        return {"overall": 0.5, "per_judge": {}, "n_decided": 0}

    def first_won(r: PairwiseResult) -> int:
        first_item = r.item_b_id if r.position_swapped else r.item_a_id
        winner_id = r.item_a_id if r.winner == "a" else r.item_b_id
        return int(winner_id == first_item)

    overall = float(np.mean([first_won(r) for r in decided]))
    per_judge: dict[str, float] = {}
    for judge in sorted({r.judge for r in decided}):
        jr = [r for r in decided if r.judge == judge]
        per_judge[judge] = float(np.mean([first_won(r) for r in jr]))

    return {"overall": overall, "per_judge": per_judge, "n_decided": len(decided)}


def swap_consistency(results: list[PairwiseResult]) -> dict:
    """For pairs judged in both orders, fraction with the same winner (higher = less position bias)."""
    by_key: dict[tuple, list[PairwiseResult]] = {}
    for r in results:
        key = tuple(sorted((r.item_a_id, r.item_b_id))) + (r.judge,)
        by_key.setdefault(key, []).append(r)

    consistent = 0
    total = 0
    for group in by_key.values():
        if len(group) < 2:
            continue
        winners = set()
        for r in group:
            if r.winner == "tie":
                continue
            winner_id = r.item_a_id if r.winner == "a" else r.item_b_id
            winners.add(winner_id)
        if len(winners) == 1:
            consistent += 1
        total += 1

    return {
        "consistency": consistent / total if total else 0.0,
        "pairs_compared": total,
        "consistent": consistent,
    }


def length_bias_correlation(lengths: dict[str, int], ratings: dict[str, float]) -> dict:
    """Spearman correlation between text length and judge rating (length bias)."""
    ids = [i for i in ratings if i in lengths]
    if len(ids) < 3:
        return {"spearman": 0.0, "p_value": 1.0, "n": len(ids)}
    corr, p = stats.spearmanr([lengths[i] for i in ids], [ratings[i] for i in ids])
    if np.isnan(corr):
        corr, p = 0.0, 1.0
    return {"spearman": float(corr), "p_value": float(p), "n": len(ids)}


def length_controlled_ratings(ratings: dict[str, float], lengths: dict[str, int]) -> dict[str, float]:
    """Regress rating on log(length) and return residuals + grand mean (length-controlled score)."""
    ids = [i for i in ratings if i in lengths and lengths[i] > 0]
    if len(ids) < 3:
        return dict(ratings)

    x = np.log(np.array([lengths[i] for i in ids], dtype=float))
    y = np.array([ratings[i] for i in ids], dtype=float)
    if float(np.max(x) - np.min(x)) == 0.0:
        return dict(ratings)
    slope, intercept = np.polyfit(x, y, 1)
    controlled = {i: float(ratings[i] - slope * np.log(lengths[i])) for i in ids}
    grand = float(np.mean(list(controlled.values())))
    return {i: controlled[i] - grand + float(np.mean(y)) for i in ids}


def blind_texts(texts: dict[str, str], seed: int = 0) -> tuple[dict[str, str], dict[str, str]]:
    """Anonymize items to neutral labels with a deterministic shuffle, hiding model identity.

    Returns (mapping id -> label, relabeled texts).
    """
    ids = sorted(texts.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    labels = {item_id: chr(ord("A") + i) for i, item_id in enumerate(ids)}
    relabeled = {labels[item_id]: texts[item_id] for item_id in ids}
    return labels, relabeled


def stable_blind_id(text: str) -> str:
    """Deterministic pseudonymous id for a text (traceable without exposing content)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
