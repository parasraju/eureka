import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import yaml

from cbf.providers import build_registry_from_settings, parse_model_spec, ProviderRegistry
from cbf.benchmarks import get_benchmark, TIER_A, list_benchmarks
from cbf.storage import CallLogger, BudgetGuard, RunManifest
from cbf.providers.base import LLMRequest


class BenchmarkRunner:
    def __init__(self, config: dict, runs_dir: str = "runs"):
        self.config = config
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.registry: ProviderRegistry | None = None
        self.logger: CallLogger | None = None
        self.budget: BudgetGuard | None = None
        self.run_id: str | None = None

    async def initialize(self, model_spec: str, max_usd: float | None = None):
        self.registry = build_registry_from_settings(self.config)
        provider_name, model_name = parse_model_spec(model_spec)
        provider = self.registry.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not configured")

        budget_limit = max_usd or self.config.get("budget", {}).get("max_usd_per_run", 50.0)
        self.run_id = f"{model_spec.replace(':', '-')}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.logger = CallLogger(self.run_id, runs_dir=str(self.runs_dir))
        self.budget = BudgetGuard(budget_limit)

        self.logger.update_manifest(
            models=[model_spec],
            provider_versions={provider_name: "1.0"},
            config_hash=self._hash_config(),
            embedding_model=self.config.get("embeddings", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        )

        return provider, model_name

    def _hash_config(self) -> str:
        config_str = yaml.dump(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def run_benchmark(self, provider, model: str, benchmark_name: str, conditions: dict) -> dict:
        bench = get_benchmark(benchmark_name)
        bench_config = self.config.get("benchmarks", {}).get(benchmark_name, {})

        all_results = {}

        for condition_name, condition_cfg in conditions.items():
            run_cfg = {**bench_config, **condition_cfg}
            run_cfg["condition"] = condition_name

            raw_results = await bench.run(provider, model, run_cfg)

            for item in raw_results.items:
                if "call_id" in item:
                    call = next((c for c in self.logger.load_calls() if c.call_id == item["call_id"]), None)
                    if call:
                        continue

            for item in raw_results.items:
                if "call_id" in item and "error" not in item:
                    call_id = item["call_id"]
                    call = next((c for c in self.logger.load_calls() if c.call_id == call_id), None)
                    if not call:
                        pass

            scores = bench.score(raw_results)

            all_results[condition_name] = {
                "raw": raw_results,
                "scores": scores,
            }

        return all_results

    async def run_tier_a(self, provider, model: str, conditions: dict) -> dict:
        results = {}
        for bench_name in TIER_A:
            rprint(f"[cyan]Running {bench_name}...[/cyan]")
            bench_results = await self.run_benchmark(provider, model, bench_name, conditions)
            results[bench_name] = bench_results
        return results

    def finalize(self, status: str = "completed", error: str | None = None):
        if self.logger:
            self.logger.finalize(status=status, error=error)


async def run_benchmarks(
    model_spec: str,
    suite: str = "tier_a",
    conditions: dict | None = None,
    max_usd: float | None = None,
    dry_run: bool = False,
    config: dict | None = None,
    runs_dir: str = "runs",
) -> dict:
    if config is None:
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

    runner = BenchmarkRunner(config, runs_dir)

    if dry_run:
        run_id = f"{model_spec.replace(':', '-')}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        runner.run_id = run_id
        runner.logger = CallLogger(run_id, runs_dir=str(runs_dir))
        runner.logger.update_manifest(
            models=[model_spec],
            config_hash=runner._hash_config(),
        )
        runner.finalize(status="dry_run")
        return {"run_id": run_id, "status": "dry_run", "estimated_cost": 0.0}

    provider, model = await runner.initialize(model_spec, max_usd)

    if conditions is None:
        conditions = config.get("conditions", {})
        if not conditions:
            conditions = {
                "neutral": {"condition": "neutral", "temperature": 0.7, "num_samples": 10},
                "creative": {"condition": "creative", "temperature": 0.7, "num_samples": 10},
            }

    if suite == "tier_a":
        results = await runner.run_tier_a(provider, model, conditions)
    else:
        results = {}
        for bench_name in suite.split(","):
            bench_name = bench_name.strip()
            if bench_name in list_benchmarks():
                results[bench_name] = await runner.run_benchmark(provider, model, bench_name, conditions)

    runner.finalize(status="completed")
    return {"run_id": runner.run_id, "results": results}


def rprint(msg: str):
    print(msg)