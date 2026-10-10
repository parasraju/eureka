import numpy as np


def normalize(value: float, lo: float, hi: float) -> float:
    """Normalize a raw value to 0-100 given documented bounds. Degenerate range -> 0."""
    if hi <= lo:
        return 0.0
    clipped = min(max(value, lo), hi)
    return (clipped - lo) / (hi - lo) * 100.0


def human_percentile(score: float, human_scores: list[float]) -> float:
    """Percentile of `score` within the human distribution (0-100)."""
    arr = np.asarray(human_scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr <= score) * 100.0)


def composite(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Weighted composite of dimension scores. Default: equal weights."""
    if not scores:
        return 0.0
    if weights:
        used = {k: v for k, v in scores.items() if k in weights and weights[k] > 0}
        w = {k: weights[k] for k in used}
    else:
        used = scores
        w = dict.fromkeys(used, 1.0)
    total = sum(w.values())
    if total <= 0:
        return 0.0
    return float(sum(used[k] * w[k] for k in used) / total)
