import asyncio
import json
from pathlib import Path

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from cbf.providers import build_registry_from_settings, parse_model_spec
from cbf.runner import expand_conditions, run_benchmarks
from cbf.runner.verify import verify_run

app = typer.Typer(name="cbf", help="AI Creative Benchmark Finder")
reg_app = typer.Typer(help="Manage the discovered-benchmark registry (SQLite).")
app.add_typer(reg_app, name="registry")
console = Console()


def load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@app.command()
def ping(
    provider: str | None = typer.Option(None, "--provider", "-p", help="Specific provider to ping (openai, anthropic, google, openrouter)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to test (provider:model)"),
):
    """Test connectivity to configured providers with a real API call."""
    config = load_config()
    registry = build_registry_from_settings(config)

    if not registry.list_names():
        rprint("[red]No providers configured. Check your .env file.[/red]")
        raise typer.Exit(1)

    providers_to_test = [provider] if provider else registry.list_names()

    results = {}
    for p_name in providers_to_test:
        prov = registry.get(p_name)
        if not prov:
            results[p_name] = {"status": "error", "error": "Provider not found"}
            continue

        rprint(f"[cyan]Pinging {p_name}...[/cyan]")
        try:
            healthy = asyncio.run(prov.health_check())
            if healthy:
                models = asyncio.run(prov.list_models())
                results[p_name] = {"status": "ok", "models": models[:10]}
                rprint(f"[green]✓ {p_name} healthy[/green]")
            else:
                results[p_name] = {"status": "error", "error": "Health check failed"}
                rprint(f"[red]✗ {p_name} health check failed[/red]")
        except Exception as e:
            results[p_name] = {"status": "error", "error": str(e)}
            rprint(f"[red]✗ {p_name} error: {e}[/red]")

    if model:
        provider_name, model_name = parse_model_spec(model)
        prov = registry.get(provider_name)
        if prov:
            rprint(f"[cyan]Testing model {model}...[/cyan]")
            from cbf.providers import LLMRequest
            req = LLMRequest(model=model_name, messages=[{"role": "user", "content": "ping"}], max_tokens=5)
            try:
                resp = asyncio.run(prov.complete(req))
                cost = prov.cost_usd(resp)
                results[f"{provider_name}:{model_name}"] = {
                    "status": "ok",
                    "latency_s": resp.latency_s,
                    "cost_usd": cost,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }
                rprint(f"[green]✓ Model {model} responded in {resp.latency_s:.2f}s (${cost:.6f})[/green]")
            except Exception as e:
                results[f"{provider_name}:{model_name}"] = {"status": "error", "error": str(e)}
                rprint(f"[red]✗ Model {model} error: {e}[/red]")
        else:
            rprint(f"[red]Provider {provider_name} not configured[/red]")

    table = Table(title="Ping Results")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    for name, result in results.items():
        status = result.get("status", "unknown")
        if status == "ok":
            details = f"Models: {', '.join(result.get('models', [])[:5])}"
            if "latency_s" in result:
                details = f"Latency: {result['latency_s']:.2f}s, Cost: ${result['cost_usd']:.6f}, Tokens: {result['tokens']}"
        else:
            details = result.get("error", "Unknown error")
        table.add_row(name, status, details)

    console.print(table)

    if any(r.get("status") == "error" for r in results.values()):
        raise typer.Exit(1)


@app.command()
def discover(
    since_days: int = typer.Option(30, "--since", "-s", help="Days to look back"),
    min_score: float = typer.Option(0.3, "--min-score", help="Minimum relevance score"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json"),
    source: str = typer.Option(None, "--source", help="Specific source to search (arxiv, semantic_scholar, huggingface, github)"),
    digest: bool = typer.Option(False, "--digest", help="Print the weekly 'what's new' digest from the registry"),
    digest_days: int = typer.Option(7, "--digest-days", help="Days to look back for the digest"),
):
    """Discover creativity benchmarks from live sources."""
    config = load_config()
    disc_config = config.get("discovery", {})

    from cbf.discovery import (
        ArxivConfig,
        DiscoveryConfig,
        DiscoveryEngine,
        GitHubConfig,
        HuggingFaceConfig,
        SemanticScholarConfig,
    )

    engine_config = DiscoveryConfig(
        sources=[source] if source else disc_config.get("sources", ["arxiv", "semantic_scholar", "huggingface", "github"]),
        arxiv=ArxivConfig(
            keywords=disc_config.get("keywords", [
                "creativity benchmark LLM",
                "divergent thinking language model",
                "creative writing evaluation",
                "novelty diversity generation",
            ]),
            max_results=disc_config.get("max_results", 50),
            days_back=since_days,
        ),
        semantic_scholar=SemanticScholarConfig(
            keywords=disc_config.get("keywords", [
                "creativity benchmark LLM",
                "divergent thinking language model",
            ]),
            limit=disc_config.get("max_results", 50),
        ),
        huggingface=HuggingFaceConfig(
            dataset_tags=["creativity", "creative-writing", "divergent-thinking"],
            paper_keywords=disc_config.get("keywords", ["creativity benchmark"]),
            limit=disc_config.get("max_results", 50),
        ),
        github=GitHubConfig(
            queries=disc_config.get("keywords", ["creativity benchmark language model"]),
            limit=disc_config.get("max_results", 30),
        ),
        min_score=min_score,
    )

    rprint(f"[cyan]Discovering benchmarks from {engine_config.sources}...[/cyan]")

    async def run_discovery():
        engine = DiscoveryEngine(engine_config)
        return await engine.discover()

    benchmarks = asyncio.run(run_discovery())

    rprint(f"[green]Found {len(benchmarks)} benchmarks above score {min_score}[/green]")

    if output == "json":
        import json
        data = [{"id": b.id, "source": b.source, "title": b.title, "url": b.url, "score": b.score, "summary": b.summary[:200]} for b in benchmarks]
        print(json.dumps(data, indent=2))
    else:
        table = Table(title="Discovered Benchmarks")
        table.add_column("#", style="cyan")
        table.add_column("Source", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Title")

        for i, b in enumerate(benchmarks[:20]):
            table.add_row(str(i+1), b.source, f"{b.score:.2f}", b.title[:80])

        console.print(table)

        if len(benchmarks) > 20:
            rprint(f"[dim]... and {len(benchmarks) - 20} more[/dim]")

    from dataclasses import asdict

    from cbf.discovery import RegistryDB

    with RegistryDB() as db:
        db.upsert([asdict(b) for b in benchmarks])
        rprint(f"[green]Saved {len(benchmarks)} candidates to registry[/green]")
        if digest:
            print(db.digest(days=digest_days))


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model to test (provider:model)"),
    suite: str = typer.Option("tier_a", "--suite", help="Benchmark suite to run (tier_a, or comma-separated list)"),
    condition: str = typer.Option(None, "--condition", "-c", help="Specific condition to run (neutral, creative, etc.)"),
    num_samples: int = typer.Option(None, "--samples", "-n", help="Number of samples per condition"),
    temperature: float = typer.Option(None, "--temperature", "-t", help="Temperature for generation"),
    max_usd: float = typer.Option(None, "--max-usd", help="Maximum budget in USD"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without making calls"),
    judges: str = typer.Option(None, "--judges", help="Comma-separated judge specs (provider:model); overrides config"),
    rubric: str = typer.Option(None, "--rubric", help="Override the judge rubric text"),
):
    """Run benchmarks against a model."""
    config = load_config()

    if judges or rubric:
        existing = config.get("judges")
        judge_cfg = dict(existing) if isinstance(existing, dict) else {"models": list(existing or [])}
        if judges:
            judge_cfg["models"] = [s.strip() for s in judges.split(",") if s.strip()]
        if rubric:
            judge_cfg["rubric"] = rubric
        config["judges"] = judge_cfg

    conditions = expand_conditions(config.get("conditions", {}))
    if not conditions:
        conditions = {
            "neutral": {"condition": "neutral", "temperature": 0.7, "num_samples": 10},
            "creative": {"condition": "creative", "temperature": 0.7, "num_samples": 10},
        }

    if condition:
        matched = {name: cfg for name, cfg in conditions.items() if cfg.get("condition") == condition}
        conditions = matched or {condition: conditions.get(condition, {"condition": condition})}

    if num_samples:
        for cond in conditions.values():
            cond["num_samples"] = num_samples

    if temperature is not None:
        for cond in conditions.values():
            cond["temperature"] = temperature

    rprint(f"[cyan]Starting run for {model} with suite={suite}[/cyan]")
    rprint(f"Conditions: {list(conditions.keys())}")

    try:
        result = asyncio.run(run_benchmarks(
            model_spec=model,
            suite=suite,
            conditions=conditions,
            max_usd=max_usd,
            dry_run=dry_run,
            config=config,
        ))

        if result.get("status") == "dry_run":
            rprint(f"[yellow]Dry run completed. Run ID: {result['run_id']}[/yellow]")
            return

        rprint(f"[green]Run completed! Run ID: {result['run_id']}[/green]")

        for bench_name, bench_results in result.get("results", {}).items():
            rprint(f"\n[cyan]{bench_name}[/cyan]")
            for cond_name, cond_result in bench_results.items():
                scores = cond_result["scores"]
                rprint(f"  {cond_name}:")
                for metric, value in scores.metrics.items():
                    ci = scores.confidence_intervals.get(metric)
                    if ci:
                        rprint(f"    {metric}: {value:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})")
                    else:
                        rprint(f"    {metric}: {value:.3f}")

    except Exception as e:
        rprint(f"[red]Run failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def verify(
    run_id: str = typer.Argument(..., help="Run ID to verify"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Runs directory"),
):
    """Verify a run by recomputing scores from raw call logs."""
    rprint(f"[cyan]Verifying run {run_id}...[/cyan]")

    result = verify_run(run_id, runs_dir)

    if not result.get("success", False):
        rprint(f"[red]Verification failed: {result.get('error', 'Unknown error')}[/red]")
        raise typer.Exit(1)

    rprint("[green]Verification successful![/green]")
    rprint(f"Run ID: {result['run_id']}")
    rprint(f"Benchmark: {result['benchmark']}")
    rprint(f"Model: {result['model']}")

    if result.get("benchmark") == "dry_run":
        rprint(result.get("message", "Dry run - no API calls made"))
        return

    rprint(f"Calls match: {result['call_count_match']}")
    rprint(f"Cost match: {result['cost_match']}")
    rprint("\nComputed scores:")
    for metric, value in result["computed_scores"].items():
        rprint(f"  {metric}: {value:.3f}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to report on"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown, html, json"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Runs directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Generate a report for a completed run."""
    from cbf.report import (
        generate_html_report,
        generate_json_report,
        generate_markdown_report,
    )

    if format == "markdown":
        content = generate_markdown_report(run_id, runs_dir)
        if output:
            with open(output, "w") as f:
                f.write(content)
            rprint(f"[green]Report saved to {output}[/green]")
        else:
            print(content)
    elif format == "html":
        content = generate_html_report(run_id, runs_dir)
        if output:
            with open(output, "w") as f:
                f.write(content)
            rprint(f"[green]Report saved to {output}[/green]")
        else:
            print(content)
    elif format == "json":
        content = generate_json_report(run_id, runs_dir)
        if output:
            with open(output, "w") as f:
                json.dump(content, f, indent=2)
            rprint(f"[green]Report saved to {output}[/green]")
        else:
            print(json.dumps(content, indent=2))
    else:
        rprint(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)


@reg_app.command("list")
def registry_list(
    status: str | None = typer.Option(None, "--status", help="Filter by status (seed, candidate, adapter_drafted, approved, rejected)"),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum composite score"),
    db_path: str = typer.Option("data/registry.db", "--db", help="Registry database path"),
):
    """List discovered benchmarks from the registry."""
    from cbf.discovery import RegistryDB

    with RegistryDB(db_path) as db:
        entries = db.search(status=status, min_score=min_score)
        table = Table(title=f"Registry — {status or 'all'} ({len(entries)} entries)")
        table.add_column("Score", style="green")
        table.add_column("Status", style="cyan")
        table.add_column("Title")
        table.add_column("Source")
        for e in entries[:60]:
            table.add_row(f"{e.score:.2f}", e.status, e.title[:70], e.source)
        console.print(table)
        if len(entries) > 60:
            rprint(f"[dim]... and {len(entries) - 60} more[/dim]")


@reg_app.command("approve")
def registry_approve(
    entry_id: str = typer.Argument(..., help="Registry id of the benchmark to approve"),
    reason: str = typer.Option("human approval", "--reason", help="Approval note"),
    db_path: str = typer.Option("data/registry.db", "--db", help="Registry database path"),
):
    """Approve a benchmark so its adapter may run on a published leaderboard."""
    from cbf.discovery import RegistryDB

    with RegistryDB(db_path) as db:
        db.set_status(entry_id, "approved", reason=reason)
    rprint(f"[green]Approved {entry_id}[/green]")


@reg_app.command("status")
def registry_status(
    entry_id: str = typer.Argument(..., help="Registry id"),
    status: str = typer.Argument(..., help="seed | candidate | adapter_drafted | approved | rejected"),
    reason: str = typer.Option("", "--reason", help="Optional note"),
    db_path: str = typer.Option("data/registry.db", "--db", help="Registry database path"),
):
    """Set a benchmark's status manually."""
    from cbf.discovery import RegistryDB

    with RegistryDB(db_path) as db:
        db.set_status(entry_id, status, reason=reason)
    rprint(f"[green]{entry_id} -> {status}[/green]")


@reg_app.command("digest")
def registry_digest(
    days: int = typer.Option(7, "--days", help="Look-back window in days"),
    db_path: str = typer.Option("data/registry.db", "--db", help="Registry database path"),
):
    """Print the 'what's new' discovery digest."""
    from cbf.discovery import RegistryDB

    with RegistryDB(db_path) as db:
        print(db.digest(days=days))


@app.command()
def leaderboard(
    output: str = typer.Option(None, "--output", "-o", help="Write markdown leaderboard to a file"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Runs directory"),
):
    """Generate a leaderboard across all runs, best run per model and benchmark."""
    from pathlib import Path

    from cbf.report import generate_leaderboard, generate_leaderboard_markdown

    run_ids = [d.name for d in Path(runs_dir).iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    if not run_ids:
        rprint("[yellow]No runs found.[/yellow]")
        return

    board = generate_leaderboard(run_ids, runs_dir)
    content = generate_leaderboard_markdown(board)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        rprint(f"[green]Leaderboard saved to {output}[/green]")
    else:
        print(content)


@app.command()
def compare(
    runs: str = typer.Argument(..., help="Comma-separated run IDs"),
    output: str = typer.Option(None, "--output", "-o", help="Write markdown to a file"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Runs directory"),
):
    """Compare multiple runs side by side (CI + cost)."""
    from cbf.report import compare_runs, render_comparison_markdown

    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    content = render_comparison_markdown(compare_runs(run_ids, runs_dir))
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        rprint(f"[green]Comparison saved to {output}[/green]")
    else:
        print(content)


@app.command()
def radar(
    runs: str = typer.Argument(..., help="Comma-separated run IDs to overlay on one chart"),
    output: str = typer.Option("report/radar.png", "--output", "-o", help="PNG output path"),
    title: str = typer.Option("Creativity Profile", "--title", help="Chart title"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Runs directory"),
):
    """Render a radar chart of normalized creativity dimensions for one or more runs."""
    from cbf.report import creativity_profile, radar_chart

    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    profiles = [creativity_profile(run_id, runs_dir) for run_id in run_ids]
    models = {}
    for p in profiles:
        if "error" in p:
            rprint(f"[red]{p['error']}[/red]")
            continue
        models[p["model"]] = p["dimensions"]

    if not models:
        rprint("[red]No comparable profiles.[/red]")
        raise typer.Exit(1)

    if len({frozenset(m.keys()) for m in models.values()}) != 1:
        rprint("[red]Runs cover different benchmark dimensions; cannot overlay on one radar.[/red]")
        raise typer.Exit(1)

    path = radar_chart(models, output, title=title)
    rprint(f"[green]Radar chart saved to {path}[/green]")


if __name__ == "__main__":
    app()
