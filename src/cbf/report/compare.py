"""Run-to-run comparison and normalized creativity profiles (plan section 4.5)."""

from datetime import UTC, datetime

from cbf.metrics import composite, normalize
from cbf.report.generator import compute_scores_from_run, load_run_results

# Documented normalization bounds per dimension. Raw values outside [lo, hi] clip.
# These are fixed so scores stay comparable across runs and models.
DIMENSION_BOUNDS = {
    "divergent_thinking": (0.0, 100.0),   # DAT already scaled by *100
    "fluency": (0.0, 10.0),               # AUT uses returned
    "flexibility": (0.0, 20.0),           # AUT novelty count heuristic
    "originality": (0.0, 100.0),          # AUT pairwise distance * 100
    "diversity": (0.0, 1.0),              # mean cosine distance
    "chain_validity": (0.0, 1.0),         # associative validity rate
    "repurposing": (0.0, 1.0),            # affordance feasibility
    "writing_elo": (800.0, 1600.0),       # Elo anchor range
}

# metric name (as emitted by each benchmark score()) -> dimension + raw transform
_METRIC_DIMENSIONS = {
    "dat": ("dat_score", "divergent_thinking", lambda v: v),
    "aut": ("fluency", "fluency", lambda v: v),
    "diversity": ("diversity_score", "diversity", lambda v: v),   # distance 0-1, normalized against bounds
    "associative": ("chain_validity", "chain_validity", lambda v: v),
    "affordance": ("feasibility", "repurposing", lambda v: v),
    "writing_elo": ("elo_rating", "writing_elo", lambda v: v),
}


def creativity_profile(run_id: str, runs_dir: str = "runs", weights: dict[str, float] | None = None) -> dict:
    """Normalize a run's per-benchmark scores into a 0-100 creativity profile + composite.

    Every normalization uses DIMENSION_BOUNDS (documented above); nothing is inferred
    from a single run's range, so profiles are comparable across runs.
    """
    profile: dict[str, float] = {}
    sources: dict[str, str] = {}
    details: dict[str, dict] = {}

    run_data = load_run_results(run_id, runs_dir)
    if "error" in run_data:
        return {"error": run_data["error"]}
    scores = compute_scores_from_run(run_data)
    if "error" in scores:
        return {"error": scores["error"]}

    bench = scores["benchmark"]
    mapping = _METRIC_DIMENSIONS.get(bench)
    if mapping:
        metric_name, dim, transform = mapping
        raw = scores["metrics"].get(metric_name)
        if raw is not None:
            lo, hi = DIMENSION_BOUNDS[dim]
            value = transform(raw)
            scaled = normalize(value, lo, hi)
            profile[dim] = round(float(scaled), 2)
            sources[dim] = f"{bench}.{metric_name}"
            details[dim] = {
                "metric": metric_name,
                "raw": float(raw),
                "bounds": {"lo": lo, "hi": hi},
                "normalized": profile[dim],
            }

    overall = round(composite(profile, weights), 2) if profile else 0.0
    return {
        "run_id": run_id,
        "model": scores["model"],
        "dimensions": profile,
        "composite": overall,
        "sources": sources,
        "details": details,
        "normalization": "DIMENSION_BOUNDS (documented fixed ranges)",
        "cost_usd": scores["run_info"]["total_cost_usd"],
        "calls": scores["run_info"]["total_calls"],
    }


def compare_runs(run_ids: list[str], runs_dir: str = "runs") -> dict:
    """Compare multiple runs side by side, grouped by benchmark, with CI and cost."""
    rows: dict[str, list[dict]] = {}
    for run_id in run_ids:
        run_data = load_run_results(run_id, runs_dir)
        if "error" in run_data:
            continue
        scores = compute_scores_from_run(run_data)
        if "error" in scores:
            continue

        bench = scores["benchmark"]
        rows.setdefault(bench, []).append({
            "run_id": run_id,
            "model": scores["model"],
            "metrics": scores["metrics"],
            "confidence_intervals": scores["confidence_intervals"],
            "cost_usd": scores["run_info"]["total_cost_usd"],
            "calls": scores["run_info"]["total_calls"],
            "started_utc": scores["run_info"]["started_utc"],
        })

    return {"rows": rows, "compared": len(run_ids)}


def render_comparison_markdown(comparison: dict) -> str:
    lines = [
        "# Run Comparison",
        "",
        f"*Compared {comparison['compared']} runs at {datetime.now(UTC).isoformat()}Z*",
        "",
    ]
    if not comparison["rows"]:
        lines.append("_No comparable runs found._")
        return "\n".join(lines)

    for bench, runs in sorted(comparison["rows"].items()):
        lines.append(f"## {bench}")
        lines.append("")
        all_metrics = sorted({m for r in runs for m in r["metrics"]})
        header = "| Run | Model | " + " | ".join(all_metrics) + " | Cost |"
        sep = "|-----|-------|" + "|".join(["-------------"] * len(all_metrics)) + "|------|"
        lines += [header, sep]
        for r in runs:
            vals = []
            for m in all_metrics:
                ci = r["confidence_intervals"].get(m)
                v = r["metrics"][m]
                if ci:
                    vals.append(f"{v:.3f} ({ci[0]:.2f}-{ci[1]:.2f})")
                else:
                    vals.append(f"{v:.3f}")
            lines.append(f"| {r['run_id']} | {r['model']} | " + " | ".join(vals) + f" | ${r['cost_usd']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def render_profiles_markdown(profiles: list[dict]) -> str:
    lines = ["# Creativity Profiles", ""]
    if not profiles:
        lines.append("_No profiles._")
        return "\n".join(lines)

    dims: list[str] = []
    for p in profiles:
        for d in p.get("dimensions", {}):
            if d not in dims:
                dims.append(d)

    lines.append("| Model | " + " | ".join(dims) + " | Composite | Cost |")
    lines.append("|-------|" + "|".join(["----"] * len(dims)) + "|-----------|------|")
    for p in profiles:
        cells = [f"{p['dimensions'].get(d, '-'):.2f}" if isinstance(p["dimensions"].get(d), float) else "-" for d in dims]
        lines.append(f"| {p['model']} | " + " | ".join(cells) + f" | {p['composite']:.2f} | ${p['cost_usd']:.4f} |")
    lines.append("")
    lines.append("_All dimension scores normalized to 0-100 with fixed documented bounds._")
    return "\n".join(lines)
