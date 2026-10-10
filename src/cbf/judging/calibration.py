"""Judge calibration: leniency/severity correction inspired by Judge Response Theory (AGC-Bench).

Each judge gets a severity parameter; item scores are corrected by removing it before
aggregation. Positive severity = judge is systematically harsher than the panel.
"""

import numpy as np

from cbf.judging.panel import PairwiseResult, compute_elo


def judge_severities(results: list[PairwiseResult]) -> dict[str, float]:
    """Severity per judge = (judge's mean item rating) - (panel grand mean).

    Uses per-judge Elo over the same item set so the scales are comparable.
    """
    judges = sorted({r.judge for r in results})
    if not judges:
        return {}

    per_judge_ratings: dict[str, dict[str, float]] = {}
    for judge in judges:
        jr = [r for r in results if r.judge == judge]
        per_judge_ratings[judge] = compute_elo(jr)

    all_means = [float(np.mean(list(v.values()))) for v in per_judge_ratings.values()]
    grand = float(np.mean(all_means))
    return {j: float(np.mean(list(v.values()))) - grand for j, v in per_judge_ratings.items()}


def apply_severity_correction(
    ratings_by_judge: dict[str, dict[str, float]],
    severities: dict[str, float] | None = None,
) -> dict[str, float]:
    """Remove each judge's severity, then average across judges per item."""
    if not ratings_by_judge:
        return {}
    if severities is None:
        severities = judge_severities_from_ratings(ratings_by_judge)

    corrected: dict[str, list[float]] = {}
    for judge, ratings in ratings_by_judge.items():
        sev = severities.get(judge, 0.0)
        for item, rating in ratings.items():
            corrected.setdefault(item, []).append(rating - sev)

    return {item: float(np.mean(vals)) for item, vals in corrected.items()}


def judge_severities_from_ratings(ratings_by_judge: dict[str, dict[str, float]]) -> dict[str, float]:
    """Severity from arbitrary per-judge rating dicts (mean deviation from grand mean)."""
    if not ratings_by_judge:
        return {}
    means = {j: float(np.mean(list(r.values()))) for j, r in ratings_by_judge.items() if r}
    if not means:
        return {}
    grand = float(np.mean(list(means.values())))
    return {j: m - grand for j, m in means.items()}


def estimate_ability_and_severity(
    results: list[PairwiseResult],
    n_iter: int = 500,
    lr: float = 0.05,
    reg: float = 1e-3,
) -> dict:
    """Jointly estimate item ability (theta) and judge severity (beta) by gradient descent
    on P(judge prefers i over k) = sigmoid(theta_i - theta_k + beta_j).

    beta_j > 0 means the judge leans toward the first-listed item (position-style bias).
    """
    items = sorted({r.item_a_id for r in results} | {r.item_b_id for r in results})
    judges = sorted({r.judge for r in results})
    if not items or not judges:
        return {"ability": {}, "severity": {}, "log_loss": 0.0}

    item_idx = {it: i for i, it in enumerate(items)}
    judge_idx = {j: i for i, j in enumerate(judges)}

    theta = np.zeros(len(items))
    beta = np.zeros(len(judges))

    pairs = []
    for r in results:
        if r.winner == "tie":
            continue
        i = item_idx[r.item_a_id if r.winner == "a" else r.item_b_id]
        k = item_idx[r.item_b_id if r.winner == "a" else r.item_a_id]
        pairs.append((i, k, judge_idx[r.judge]))

    if not pairs:
        return {"ability": dict.fromkeys(items, 0.0), "severity": dict.fromkeys(judges, 0.0), "log_loss": 0.0}

    idx_i = np.array([p[0] for p in pairs])
    idx_k = np.array([p[1] for p in pairs])
    idx_j = np.array([p[2] for p in pairs])

    for _ in range(n_iter):
        margin = theta[idx_i] - theta[idx_k] + beta[idx_j]
        p = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
        err = p - 1.0

        grad_theta = np.zeros(len(items))
        grad_beta = np.zeros(len(judges))
        np.add.at(grad_theta, idx_i, err)
        np.add.at(grad_theta, idx_k, -err)
        np.add.at(grad_beta, idx_j, err)

        grad_theta -= reg * theta
        grad_beta -= reg * beta

        theta -= lr * grad_theta / len(pairs)
        beta -= lr * grad_beta / len(pairs)

    final_margin = theta[idx_i] - theta[idx_k] + beta[idx_j]
    p_final = 1.0 / (1.0 + np.exp(-np.clip(final_margin, -30, 30)))
    eps = 1e-9
    log_loss = float(-np.mean(np.log(p_final + eps)))

    theta = theta - theta.mean()
    return {
        "ability": {items[i]: float(theta[i]) for i in range(len(items))},
        "severity": {judges[j]: float(beta[j]) for j in range(len(judges))},
        "log_loss": log_loss,
    }


def calibrated_panel_scores(results: list[PairwiseResult]) -> dict:
    """Severity-corrected item scores plus diagnostics for a panel run."""
    estimate = estimate_ability_and_severity(results)
    severities = estimate["severity"]
    per_judge: dict[str, dict[str, float]] = {}
    for judge in sorted({r.judge for r in results}):
        per_judge[judge] = compute_elo([r for r in results if r.judge == judge])

    corrected = apply_severity_correction(per_judge, judge_severities_from_ratings(per_judge))
    return {
        "scores": corrected,
        "ability": estimate["ability"],
        "severity": severities,
        "log_loss": estimate["log_loss"],
        "reliable": len({r.judge for r in results}) >= 2,
    }
