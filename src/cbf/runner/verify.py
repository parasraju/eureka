import json
import math
from pathlib import Path

from cbf.benchmarks import get_benchmark
from cbf.benchmarks.base import RawResults
from cbf.storage import load_calls, load_manifest

MISSING_RESULTS_TEMPLATE = (
    "Run {run_id} has no persisted results (results.jsonl). "
    "Re-run with the current version."
)


def _load_persisted_results(
    run_id: str,
    runs_dir: str,
    result_filename: str = "results.jsonl",
) -> list[dict] | None:
    results_file = Path(runs_dir) / run_id / result_filename
    if not results_file.exists():
        return None

    results: list[dict] = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _default_model(manifest) -> str:
    return manifest.models[0] if manifest.models else "unknown"


def _metrics_match(stored, computed) -> bool:
    try:
        return math.isclose(float(stored), float(computed), rel_tol=1e-6, abs_tol=1e-9)
    except (TypeError, ValueError):
        return stored == computed


def verify_run(run_id: str, runs_dir: str = "runs") -> dict:
    manifest = load_manifest(run_id, runs_dir)
    if not manifest:
        return {"success": False, "error": f"Run {run_id} not found", "run_id": run_id}

    results = _load_persisted_results(
        run_id,
        runs_dir,
        getattr(manifest, "result_file", None) or "results.jsonl",
    )
    if results is None:
        return {
            "success": False,
            "error": MISSING_RESULTS_TEMPLATE.format(run_id=run_id),
            "run_id": run_id,
        }
    if not results:
        return {
            "success": False,
            "error": MISSING_RESULTS_TEMPLATE.format(run_id=run_id),
            "run_id": run_id,
        }

    calls = load_calls(run_id, runs_dir)
    calls_by_id = {call.call_id: call for call in calls}

    mismatches: list[dict] = []
    missing_call_ids: list[dict] = []
    recompute_errors: list[dict] = []
    computed_entries: list[dict] = []
    benchmarks: list[str] = []
    models: list[str] = []

    for entry in results:
        scores = entry.get("scores") or {}
        benchmark_id = entry.get("benchmark") or scores.get("benchmark_id", "")
        condition = entry.get("condition", "")
        model = entry.get("model") or scores.get("model") or _default_model(manifest)

        if benchmark_id and benchmark_id not in benchmarks:
            benchmarks.append(benchmark_id)
        if model and model not in models:
            models.append(model)

        raw_dict = entry.get("raw") or {}
        for item in raw_dict.get("items") or []:
            call_id = item.get("call_id")
            if call_id and call_id not in calls_by_id:
                missing_call_ids.append({
                    "benchmark": benchmark_id,
                    "condition": condition,
                    "call_id": call_id,
                })

        try:
            benchmark = get_benchmark(benchmark_id)
            raw_results = RawResults(
                benchmark_id=raw_dict.get("benchmark_id", benchmark_id),
                model=raw_dict.get("model", model),
                items=list(raw_dict.get("items") or []),
                call_ids=list(raw_dict.get("call_ids") or []),
                metadata=dict(raw_dict.get("metadata") or {}),
            )
            recomputed = benchmark.score(raw_results)
        except Exception as exc:
            recompute_errors.append({
                "benchmark": benchmark_id,
                "condition": condition,
                "recompute_error": f"{type(exc).__name__}: {exc}",
            })
            continue

        computed_metrics = dict(recomputed.metrics)
        computed_entries.append({
            "benchmark": benchmark_id,
            "condition": condition,
            "metrics": computed_metrics,
        })

        stored_metrics = dict(scores.get("metrics") or {})
        for metric in sorted(set(stored_metrics) | set(computed_metrics)):
            if metric not in computed_metrics:
                mismatches.append({
                    "benchmark": benchmark_id,
                    "condition": condition,
                    "metric": metric,
                    "stored": stored_metrics[metric],
                    "computed": None,
                    "reason": "metric_missing_from_recompute",
                })
            elif metric not in stored_metrics:
                mismatches.append({
                    "benchmark": benchmark_id,
                    "condition": condition,
                    "metric": metric,
                    "stored": None,
                    "computed": computed_metrics[metric],
                    "reason": "metric_missing_from_results",
                })
            elif not _metrics_match(stored_metrics[metric], computed_metrics[metric]):
                mismatches.append({
                    "benchmark": benchmark_id,
                    "condition": condition,
                    "metric": metric,
                    "stored": stored_metrics[metric],
                    "computed": computed_metrics[metric],
                    "reason": "value_mismatch",
                })

    call_count_match = len(calls) == manifest.total_calls
    cost_match = math.isclose(
        sum(call.cost_usd for call in calls),
        manifest.total_cost_usd,
        rel_tol=1e-6,
        abs_tol=1e-9,
    )

    success = (
        not mismatches
        and not missing_call_ids
        and not recompute_errors
        and call_count_match
        and cost_match
    )

    if len(computed_entries) == 1:
        computed_scores = computed_entries[0]["metrics"]
    else:
        computed_scores = {
            f"{entry['benchmark']}/{entry['condition']}/{metric}": value
            for entry in computed_entries
            for metric, value in entry["metrics"].items()
        }

    verification = {
        "success": success,
        "run_id": run_id,
        "benchmark": ", ".join(benchmarks) if benchmarks else None,
        "model": ", ".join(models) if models else _default_model(manifest),
        "verified_results": len(computed_entries),
        "mismatches": mismatches,
        "missing_call_ids": missing_call_ids,
        "recompute_errors": recompute_errors,
        "call_count_match": call_count_match,
        "cost_match": cost_match,
        "computed_scores": computed_scores,
        "stored_manifest": {
            "total_cost_usd": manifest.total_cost_usd,
            "total_calls": manifest.total_calls,
            "status": manifest.status,
        },
    }

    if not success:
        problems = []
        if mismatches:
            problems.append(f"{len(mismatches)} metric mismatch(es)")
        if missing_call_ids:
            problems.append(f"{len(missing_call_ids)} missing call id(s)")
        if recompute_errors:
            problems.append(f"{len(recompute_errors)} recompute error(s)")
        if not call_count_match:
            problems.append("call count does not match calls.jsonl")
        if not cost_match:
            problems.append("cost does not match calls.jsonl")
        verification["error"] = "Verification failed: " + "; ".join(problems)

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
