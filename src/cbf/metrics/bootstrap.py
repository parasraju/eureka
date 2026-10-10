import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval. Deterministic for a fixed seed."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_stats = statistic(arr[idx], axis=1)
    low = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (low, high)


def t_ci(values: list[float] | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Student-t confidence interval (legacy behaviour used by early suites)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)
    if _stats is None:  # pragma: no cover
        m = float(np.mean(arr))
        return (m, m)
    mean = float(np.mean(arr))
    sem = _stats.sem(arr)
    low, high = _stats.t.interval(1 - alpha, arr.size - 1, loc=mean, scale=sem)
    return (float(low), float(high))


def mean_ci(
    values: list[float] | np.ndarray,
    method: str = "bootstrap",
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if method == "t":
        return t_ci(values, alpha=alpha)
    return bootstrap_ci(values, alpha=alpha, seed=seed)


def bootstrap_comparison(
    a: list[float] | np.ndarray,
    b: list[float] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap the difference mean(a) - mean(b) (used for run-to-run/model comparison)."""
    ra = np.asarray(a, dtype=float)
    rb = np.asarray(b, dtype=float)
    ra = ra[~np.isnan(ra)]
    rb = rb[~np.isnan(rb)]
    if ra.size == 0 or rb.size == 0:
        return {"diff": 0.0, "ci": (0.0, 0.0), "significant": False}

    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, ra.size, size=(n_boot, ra.size))
    idx_b = rng.integers(0, rb.size, size=(n_boot, rb.size))
    diffs = ra[idx_a].mean(axis=1) - rb[idx_b].mean(axis=1)
    low = float(np.percentile(diffs, 100 * (alpha / 2)))
    high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "diff": float(ra.mean() - rb.mean()),
        "ci": (low, high),
        "significant": not (low <= 0.0 <= high),
    }
