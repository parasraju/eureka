from pathlib import Path

from cbf.benchmarks import get_benchmark
from cbf.storage import load_calls, load_manifest


def verify_run(run_id: str, runs_dir: str = "runs") -> dict:
    manifest = load_manifest(run_id, runs_dir)
    if not manifest:
        return {"success": False, "error": f"Run {run_id} not found"}

    if manifest.status == "dry_run":
        return {
            "success": True,
            "run_id": run_id,
            "benchmark": "dry_run",
            "model": manifest.models[0] if manifest.models else "unknown",
            "message": "Dry run - no API calls made",
            "stored_manifest": {
                "total_cost_usd": manifest.total_cost_usd,
                "total_calls": manifest.total_calls,
                "status": manifest.status,
            },
        }

    calls = load_calls(run_id, runs_dir)
    if not calls:
        return {"success": False, "error": f"No call logs found for run {run_id}"}

    benchmark_id = manifest.config_hash.split("_")[0] if "_" in manifest.config_hash else "dat"
    try:
        bench = get_benchmark(benchmark_id)
    except ValueError:
        return {"success": False, "error": f"Unknown benchmark: {benchmark_id}"}

    raw_results = {
        "benchmark_id": benchmark_id,
        "model": manifest.models[0] if manifest.models else "unknown",
        "items": [],
        "call_ids": [c.call_id for c in calls],
        "metadata": {},
    }

    for call in calls:
        if call.error:
            raw_results["items"].append({
                "call_id": call.call_id,
                "error": call.error,
            })
            continue

        raw_results["items"].append({
            "call_id": call.call_id,
            "response": call.response.get("text", "") if call.response else "",
            "latency_s": call.latency_s,
            "tokens": {"input": call.input_tokens, "output": call.output_tokens},
        })

    scores = bench.score(raw_results)

    verification = {
        "success": True,
        "run_id": run_id,
        "benchmark": benchmark_id,
        "model": raw_results["model"],
        "computed_scores": scores.metrics,
        "stored_manifest": {
            "total_cost_usd": manifest.total_cost_usd,
            "total_calls": manifest.total_calls,
            "status": manifest.status,
        },
        "call_count_match": len(calls) == manifest.total_calls,
        "cost_match": abs(sum(c.cost_usd for c in calls) - manifest.total_cost_usd) < 0.001,
    }

    return verification


def verify_all_runs(runs_dir: str = "runs") -> list[dict]:
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []

    results = []
    for run_dir in runs_path.iterdir():
        if run_dir.is_dir() and (run_dir / "manifest.json").exists():
            result = verify_run(run_dir.name, runs_dir)
            results.append(result)
    return results
