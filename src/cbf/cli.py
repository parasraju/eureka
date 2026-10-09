import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from cbf.providers import build_registry_from_settings, parse_model_spec
from cbf.storage import CallLogger, BudgetGuard, RunManifest, load_manifest, load_calls
from cbf.benchmarks import get_benchmark, TIER_A, list_benchmarks
from cbf.runner import run_benchmarks
from cbf.runner.verify import verify_run
import yaml

app = typer.Typer(name="cbf", help="AI Creative Benchmark Finder")
console = Console()


def load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@app.command()
def ping(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Specific provider to ping (openai, anthropic, google, openrouter)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to test (provider:model)"),
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
):
    """Discover creativity benchmarks from live sources."""
    config = load_config()
    disc_config = config.get("discovery", {})

    from cbf.discovery import DiscoveryEngine, DiscoveryConfig, ArxivConfig, SemanticScholarConfig, HuggingFaceConfig, GitHubConfig

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


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model to test (provider:model)"),
    suite: str = typer.Option("tier_a", "--suite", help="Benchmark suite to run (tier_a, or comma-separated list)"),
    condition: str = typer.Option(None, "--condition", "-c", help="Specific condition to run (neutral, creative, etc.)"),
    num_samples: int = typer.Option(None, "--samples", "-n", help="Number of samples per condition"),
    temperature: float = typer.Option(None, "--temperature", "-t", help="Temperature for generation"),
    max_usd: float = typer.Option(None, "--max-usd", help="Maximum budget in USD"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without making calls"),
):
    """Run benchmarks against a model."""
    config = load_config()

    conditions = config.get("conditions", {})
    if not conditions:
        conditions = {
            "neutral": {"condition": "neutral", "temperature": 0.7, "num_samples": 10},
            "creative": {"condition": "creative", "temperature": 0.7, "num_samples": 10},
        }

    if condition:
        conditions = {condition: conditions.get(condition, {"condition": condition})}

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
        raise typer.Exit(1)


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

    rprint(f"[green]Verification successful![/green]")
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
    from cbf.report import generate_markdown_report, generate_html_report, generate_json_report, save_report

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


if __name__ == "__main__":
    app()