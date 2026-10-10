import json
from datetime import datetime
from typing import Any

from jinja2 import Environment, select_autoescape

from cbf.benchmarks import get_benchmark
from cbf.benchmarks.base import RawResults
from cbf.storage import load_calls, load_manifest


def load_run_results(run_id: str, runs_dir: str = "runs") -> dict[str, Any]:
    manifest = load_manifest(run_id, runs_dir)
    if not manifest:
        return {"error": f"Run {run_id} not found"}

    calls = load_calls(run_id, runs_dir)

    benchmark_id = manifest.config_hash.split("_")[0] if "_" in manifest.config_hash else "dat"
    try:
        bench = get_benchmark(benchmark_id)
    except ValueError:
        bench = None

    return {
        "manifest": manifest,
        "calls": calls,
        "benchmark_id": benchmark_id,
        "benchmark": bench,
    }


def compute_scores_from_run(run_data: dict) -> dict[str, Any]:
    manifest = run_data["manifest"]
    calls = run_data["calls"]
    benchmark = run_data["benchmark"]

    if not benchmark:
        return {"error": "Unknown benchmark"}

    items = []
    for call in calls:
        if call.error:
            items.append({
                "call_id": call.call_id,
                "error": call.error,
            })
            continue

        items.append({
            "call_id": call.call_id,
            "response": call.response.get("text", "") if call.response else "",
            "latency_s": call.latency_s,
            "tokens": {"input": call.input_tokens, "output": call.output_tokens},
        })

    raw_results = RawResults(
        benchmark_id=run_data["benchmark_id"],
        model=manifest.models[0] if manifest.models else "unknown",
        items=items,
        call_ids=[c.call_id for c in calls],
        metadata={},
    )

    scores = benchmark.score(raw_results)

    return {
        "model": raw_results.model,
        "benchmark": run_data["benchmark_id"],
        "metrics": scores.metrics,
        "confidence_intervals": scores.confidence_intervals,
        "metadata": scores.metadata,
        "run_info": {
            "run_id": manifest.run_id,
            "total_cost_usd": manifest.total_cost_usd,
            "total_calls": manifest.total_calls,
            "started_utc": manifest.started_utc,
            "ended_utc": manifest.ended_utc,
        },
    }


def generate_leaderboard(run_ids: list[str], runs_dir: str = "runs") -> dict[str, Any]:
    all_results = {}

    for run_id in run_ids:
        run_data = load_run_results(run_id, runs_dir)
        if "error" in run_data:
            continue

        scores = compute_scores_from_run(run_data)
        if "error" in scores:
            continue

        model = scores["model"]
        benchmark = scores["benchmark"]

        if model not in all_results:
            all_results[model] = {}

        all_results[model][benchmark] = scores

    return all_results


def generate_markdown_report(run_id: str, runs_dir: str = "runs") -> str:
    run_data = load_run_results(run_id, runs_dir)
    if "error" in run_data:
        return f"# Error\n\nRun {run_id} not found."

    scores = compute_scores_from_run(run_data)
    if "error" in scores:
        return f"# Error\n\n{scores['error']}"

    manifest = run_data["manifest"]

    lines = [
        "# Creativity Benchmark Report",
        "",
        f"**Run ID:** {manifest.run_id}",
        f"**Model:** {scores['model']}",
        f"**Benchmark:** {scores['benchmark']}",
        f"**Status:** {manifest.status}",
        f"**Started:** {manifest.started_utc}",
        f"**Ended:** {manifest.ended_utc or 'N/A'}",
        f"**Total Cost:** ${manifest.total_cost_usd:.4f}",
        f"**Total API Calls:** {manifest.total_calls}",
        "",
        "## Scores",
        "",
    ]

    for metric, value in scores["metrics"].items():
        ci = scores["confidence_intervals"].get(metric)
        if ci:
            lines.append(f"- **{metric}**: {value:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f})")
        else:
            lines.append(f"- **{metric}**: {value:.3f}")

    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")

    if scores.get("metadata"):
        for key, value in scores["metadata"].items():
            if key != "individual_scores":
                lines.append(f"- **{key}**: {value}")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated at {datetime.utcnow().isoformat()}Z*")

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
    <p><strong>Model:</strong> {{ model }}</p>
    <p><strong>Benchmark:</strong> {{ benchmark }}</p>
    <p><strong>Status:</strong> <span class="badge {% if status == 'completed' %}badge-success{% elif status == 'running' %}badge-warning{% else %}badge-error{% endif %}">{{ status }}</span></p>
    <p><strong>Total Cost:</strong> ${{ "%.4f"|format(total_cost) }}</p>
    <p><strong>Total API Calls:</strong> {{ total_calls }}</p>

    <h2>Scores</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>
        {% for metric, value in metrics.items() %}
        <tr>
            <td>{{ metric }}</td>
            <td class="metric">{{ "%.3f"|format(value) }}</td>
            <td>
                {% if metric in confidence_intervals %}
                {{ "%.3f"|format(confidence_intervals[metric][0]) }} - {{ "%.3f"|format(confidence_intervals[metric][1]) }}
                {% else %}
                N/A
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>

    <div class="footer">
        Report generated at {{ generated_at }}Z
    </div>
</body>
</html>
    """

    run_data = load_run_results(run_id, runs_dir)
    scores = compute_scores_from_run(run_data)

    env = Environment(autoescape=select_autoescape(['html']))
    template = env.from_string(html_template)

    return template.render(
        run_id=run_id,
        model=scores.get("model", "Unknown"),
        benchmark=scores.get("benchmark", "Unknown"),
        status=run_data["manifest"].status,
        total_cost=run_data["manifest"].total_cost_usd,
        total_calls=run_data["manifest"].total_calls,
        metrics=scores.get("metrics", {}),
        confidence_intervals=scores.get("confidence_intervals", {}),
        generated_at=datetime.utcnow().isoformat(),
    )


def generate_json_report(run_id: str, runs_dir: str = "runs") -> dict:
    run_data = load_run_results(run_id, runs_dir)
    if "error" in run_data:
        return {"error": run_data["error"]}

    scores = compute_scores_from_run(run_data)
    if "error" in scores:
        return {"error": scores["error"]}

    manifest = run_data["manifest"]

    return {
        "run_id": manifest.run_id,
        "model": scores["model"],
        "benchmark": scores["benchmark"],
        "status": manifest.status,
        "started_utc": manifest.started_utc,
        "ended_utc": manifest.ended_utc,
        "total_cost_usd": manifest.total_cost_usd,
        "total_calls": manifest.total_calls,
        "scores": {
            "metrics": scores["metrics"],
            "confidence_intervals": {k: list(v) for k, v in scores["confidence_intervals"].items()},
            "metadata": scores["metadata"],
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
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
        f"*Generated at {datetime.utcnow().isoformat()}Z*",
        "",
    ]

    all_benchmarks = set()
    for model_results in leaderboard.values():
        all_benchmarks.update(model_results.keys())

    for benchmark in sorted(all_benchmarks):
        lines.append(f"## {benchmark}")
        lines.append("")

        model_scores = []
        for model, results in leaderboard.items():
            if benchmark in results:
                scores = results[benchmark]["metrics"]
                primary_metric = list(scores.keys())[0] if scores else "score"
                model_scores.append((model, scores.get(primary_metric, 0)))

        model_scores.sort(key=lambda x: x[1], reverse=True)

        lines.append("| Rank | Model | Score |")
        lines.append("|------|-------|-------|")
        for i, (model, score) in enumerate(model_scores):
            lines.append(f"| {i+1} | {model} | {score:.3f} |")
        lines.append("")

    return "\n".join(lines)
