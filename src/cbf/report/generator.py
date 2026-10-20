import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

from cbf.storage import load_manifest

MISSING_RESULTS_TEMPLATE = (
    "Run {run_id} has no persisted results (results.jsonl). "
    "Re-run with the current version."
)


def load_run_results(run_id: str, runs_dir: str = "runs") -> dict[str, Any]:
    """Load a run's manifest and persisted ``results.jsonl`` entries.

    Old runs that predate ``results.jsonl`` are reported as errors instead of
    being re-labelled as DAT, which is what the removed ``suite`` fallback did.
    """
    manifest = load_manifest(run_id, runs_dir)
    if not manifest:
        return {"error": f"Run {run_id} not found"}

    result_filename = getattr(manifest, "result_file", None) or "results.jsonl"
    results_file = Path(runs_dir) / run_id / result_filename
    if not results_file.exists():
        return {"error": MISSING_RESULTS_TEMPLATE.format(run_id=run_id)}

    results: list[dict[str, Any]] = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    return {"manifest": manifest, "results": results, "run_id": run_id}


def collect_texts_by_model(
    run_ids: list[str],
    runs_dir: str = "runs",
    benchmark: str | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Collect unique generated texts per model from persisted runs.

    Only items that produced text are returned; failed calls cannot be judged.
    Identical texts (e.g. the same run read twice) are counted once.
    """
    texts: dict[str, list[str]] = {}
    errors: list[str] = []
    seen: set[str] = set()

    for run_id in run_ids:
        run_data = load_run_results(run_id, runs_dir)
        if "error" in run_data:
            errors.append(f"{run_id}: {run_data['error']}")
            continue
        for result in run_data["results"]:
            if benchmark and result.get("benchmark") != benchmark:
                continue
            raw = result.get("raw") or {}
            model = result.get("model") or raw.get("model") or "unknown"
            for item in raw.get("items", []):
                text = item.get("text")
                if not text or "error" in item:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                key = f"{model}\0{digest}"
                if key in seen:
                    continue
                seen.add(key)
                texts.setdefault(model, []).append(text)

    return texts, errors


def group_results_by_benchmark(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result.get("benchmark", ""), []).append(result)
    return grouped


def _model_label(models: list[str], manifest) -> str:
    unique = list(dict.fromkeys(model for model in models if model))
    if not unique:
        unique = list(dict.fromkeys(manifest.models or []))
    if not unique:
        return "unknown"
    return ", ".join(unique)


def compute_scores_from_run(run_data: dict) -> dict[str, Any]:
    """Return the persisted scores for a run without re-scoring call text."""
    if "error" in run_data:
        return {"error": run_data["error"]}

    manifest = run_data["manifest"]
    raw_entries = run_data.get("results") or []
    if not raw_entries:
        return {"error": MISSING_RESULTS_TEMPLATE.format(run_id=manifest.run_id)}

    results: list[dict[str, Any]] = []
    models: list[str] = []
    for entry in raw_entries:
        scores = entry.get("scores") or {}
        model = entry.get("model") or scores.get("model")
        if model:
            models.append(model)

        result = {
            "benchmark": entry.get("benchmark") or scores.get("benchmark_id", ""),
            "condition": entry.get("condition", ""),
            "metrics": dict(scores.get("metrics") or {}),
            "confidence_intervals": {
                metric: list(interval)
                for metric, interval in (scores.get("confidence_intervals") or {}).items()
            },
            "metadata": dict(scores.get("metadata") or {}),
        }
        if entry.get("error"):
            result["error"] = entry["error"]
        results.append(result)

    primary = results[0]
    return {
        "model": _model_label(models, manifest),
        "models": list(dict.fromkeys(model for model in models if model)) or list(manifest.models or []),
        "status": manifest.status,
        "results": results,
        # Compatibility for report/compare.py, which describes one benchmark per run.
        "benchmark": primary["benchmark"],
        "metrics": primary["metrics"],
        "confidence_intervals": primary["confidence_intervals"],
        "metadata": primary["metadata"],
        "run_info": {
            "run_id": manifest.run_id,
            "status": manifest.status,
            "models": list(manifest.models or []),
            "total_cost_usd": manifest.total_cost_usd,
            "total_calls": manifest.total_calls,
            "started_utc": manifest.started_utc,
            "ended_utc": manifest.ended_utc,
        },
    }


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.3f}"
    return str(value)


def generate_leaderboard(run_ids: list[str], runs_dir: str = "runs") -> dict[str, Any]:
    """Aggregate persisted results per model and benchmark across conditions.

    Each condition contributes one mean per metric, so conditions are averaged
    evenly and never overwrite each other.
    """
    per_condition: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

    for run_id in run_ids:
        run_data = load_run_results(run_id, runs_dir)
        if "error" in run_data:
            continue

        manifest = run_data["manifest"]
        default_model = _model_label([], manifest)
        for entry in run_data.get("results") or []:
            if entry.get("error"):
                continue

            scores = entry.get("scores") or {}
            benchmark = entry.get("benchmark") or scores.get("benchmark_id")
            if not benchmark:
                continue
            model = entry.get("model") or scores.get("model") or default_model
            condition = entry.get("condition") or "default"

            metric_values = (
                per_condition
                .setdefault(model, {})
                .setdefault(benchmark, {})
                .setdefault(condition, {})
            )
            for metric, value in (scores.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_values.setdefault(metric, []).append(float(value))

    leaderboard: dict[str, dict[str, Any]] = {}
    for model, benchmarks in per_condition.items():
        for benchmark, conditions in benchmarks.items():
            condition_means: dict[str, dict[str, float]] = {
                condition: {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                    if values
                }
                for condition, metrics in conditions.items()
            }

            metric_names: list[str] = []
            for metrics in condition_means.values():
                for metric in metrics:
                    if metric not in metric_names:
                        metric_names.append(metric)

            aggregated_metrics: dict[str, float] = {}
            for metric in metric_names:
                values = [
                    metrics[metric]
                    for metrics in condition_means.values()
                    if metric in metrics
                ]
                if values:
                    aggregated_metrics[metric] = sum(values) / len(values)

            leaderboard.setdefault(model, {})[benchmark] = {
                "metrics": aggregated_metrics,
                "conditions": sorted(condition_means),
            }

    return leaderboard


def generate_markdown_report(run_id: str, runs_dir: str = "runs") -> str:
    run_data = load_run_results(run_id, runs_dir)
    if "error" in run_data:
        return f"# Error\n\n{run_data['error']}"

    scores = compute_scores_from_run(run_data)
    if "error" in scores:
        return f"# Error\n\n{scores['error']}"

    manifest = run_data["manifest"]
    run_info = scores["run_info"]

    lines = [
        "# Creativity Benchmark Report",
        "",
        f"**Run ID:** {run_info['run_id']}",
        f"**Model(s):** {scores['model']}",
        f"**Status:** {scores['status']}",
        f"**Started:** {run_info['started_utc']}",
        f"**Ended:** {run_info['ended_utc'] or 'N/A'}",
        f"**Total Cost:** ${manifest.total_cost_usd:.4f}",
        f"**Total API Calls:** {manifest.total_calls}",
        "",
    ]

    for benchmark, entries in group_results_by_benchmark(scores["results"]).items():
        for entry in entries:
            condition = entry.get("condition") or "default"
            lines.append(f"## {benchmark} / {condition}")
            lines.append("")

            for metric, value in entry["metrics"].items():
                ci = entry["confidence_intervals"].get(metric)
                if ci and len(ci) >= 2:
                    lines.append(
                        f"- **{metric}**: {_format_number(value)} "
                        f"(95% CI: {_format_number(ci[0])} - {_format_number(ci[1])})"
                    )
                else:
                    lines.append(f"- **{metric}**: {_format_number(value)}")

            if entry.get("error"):
                lines.append(f"- **error**: {entry['error']}")

            metadata = {
                key: value
                for key, value in entry.get("metadata", {}).items()
                if key != "individual_scores"
            }
            if metadata:
                lines.append("")
                lines.append("**Metadata:**")
                lines.append("")
                for key, value in metadata.items():
                    lines.append(f"- **{key}**: {value}")

            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated at {datetime.now(UTC).isoformat()}*")

    return "\n".join(lines)


def generate_html_report(run_id: str, runs_dir: str = "runs") -> str:
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Creativity Benchmark Report - {{ run_id }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
        h1, h2 { color: #1a1a2e; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #eee; }
        th { background-color: #f5f5f5; }
        .metric { font-family: monospace; }
        .badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-error { background: #f8d7da; color: #721c24; }
        .footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; color: #666; font-size: 0.875rem; }
    </style>
</head>
<body>
    <h1>Creativity Benchmark Report</h1>
    <p><strong>Run ID:</strong> {{ run_id }}</p>
    <p><strong>Model(s):</strong> {{ model }}</p>
    <p><strong>Status:</strong> <span class="badge {% if status == 'completed' %}badge-success{% elif status == 'running' %}badge-warning{% else %}badge-error{% endif %}">{{ status }}</span></p>
    <p><strong>Total Cost:</strong> ${{ "%.4f"|format(total_cost) }}</p>
    <p><strong>Total API Calls:</strong> {{ total_calls }}</p>

    {% for result in results %}
    <h2>{{ result.benchmark }} / {{ result.condition or 'default' }}</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>
        {% for metric, value in result.metrics.items() %}
        {% set ci = result.confidence_intervals.get(metric) %}
        <tr>
            <td>{{ metric }}</td>
            <td class="metric">{{ "%.3f"|format(value) }}</td>
            <td>
                {% if ci and ci|length >= 2 %}
                {{ "%.3f"|format(ci[0]) }} - {{ "%.3f"|format(ci[1]) }}
                {% else %}
                N/A
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
    {% endfor %}

    <div class="footer">
        Report generated at {{ generated_at }}Z
    </div>
</body>
</html>
    """

    run_data = load_run_results(run_id, runs_dir)
    if "error" in run_data:
        return f"<html><body><h1>Error</h1><p>{run_data['error']}</p></body></html>"

    scores = compute_scores_from_run(run_data)
    if "error" in scores:
        return f"<html><body><h1>Error</h1><p>{scores['error']}</p></body></html>"

    manifest = run_data["manifest"]
    env = Environment(autoescape=select_autoescape(["html"]))
    template = env.from_string(html_template)

    return template.render(
        run_id=manifest.run_id,
        model=scores["model"],
        status=manifest.status,
        total_cost=manifest.total_cost_usd,
        total_calls=manifest.total_calls,
        results=scores["results"],
        generated_at=datetime.now(UTC).isoformat(),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def generate_json_report(run_id: str, runs_dir: str = "runs") -> dict:
    run_data = load_run_results(run_id, runs_dir)
    if "error" in run_data:
        return {"error": run_data["error"]}

    scores = compute_scores_from_run(run_data)
    if "error" in scores:
        return {"error": scores["error"]}

    manifest = run_data["manifest"]
    results = [
        {
            "benchmark": entry["benchmark"],
            "condition": entry["condition"],
            "metrics": _json_safe(entry["metrics"]),
            "confidence_intervals": _json_safe(entry["confidence_intervals"]),
            "metadata": _json_safe(entry["metadata"]),
        }
        for entry in scores["results"]
    ]

    return {
        "run_id": manifest.run_id,
        "model": scores["model"],
        "status": manifest.status,
        "started_utc": manifest.started_utc,
        "ended_utc": manifest.ended_utc,
        "total_cost_usd": manifest.total_cost_usd,
        "total_calls": manifest.total_calls,
        "results": results,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def save_report(run_id: str, output_path: str, format: str = "markdown", runs_dir: str = "runs"):
    if format == "markdown":
        content = generate_markdown_report(run_id, runs_dir)
        with open(output_path, "w") as f:
            f.write(content)
    elif format == "html":
        content = generate_html_report(run_id, runs_dir)
        with open(output_path, "w") as f:
            f.write(content)
    elif format == "json":
        content = generate_json_report(run_id, runs_dir)
        with open(output_path, "w") as f:
            json.dump(content, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Report saved to {output_path}")


def generate_leaderboard_markdown(leaderboard: dict[str, dict]) -> str:
    lines = [
        "# Creativity Benchmark Leaderboard",
        "",
        f"*Generated at {datetime.now(UTC).isoformat()}Z*",
        "",
    ]

    all_benchmarks = set()
    for model_results in leaderboard.values():
        all_benchmarks.update(model_results.keys())

    for benchmark in sorted(all_benchmarks):
        primary_metric = "score"
        for results in leaderboard.values():
            metrics = results.get(benchmark, {}).get("metrics", {})
            if metrics:
                primary_metric = next(iter(metrics))
                break

        lines.append(f"## {benchmark}")
        lines.append("")
        lines.append(f"**Primary metric:** `{primary_metric}`")
        lines.append("")

        model_scores = []
        for model, results in leaderboard.items():
            if benchmark in results:
                entry = results[benchmark]
                metrics = entry.get("metrics", {})
                conditions = entry.get("conditions", [])
                model_scores.append((model, metrics.get(primary_metric, 0.0), len(conditions)))

        model_scores.sort(key=lambda x: x[1], reverse=True)

        lines.append(f"| Rank | Model | {primary_metric} | Conditions |")
        lines.append("|------|-------|-------|------------|")
        for i, (model, score, condition_count) in enumerate(model_scores):
            lines.append(f"| {i + 1} | {model} | {score:.3f} | {condition_count} |")
        lines.append("")

    return "\n".join(lines)
