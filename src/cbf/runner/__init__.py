import hashlib
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import yaml

from cbf.benchmarks import TIER_A, get_benchmark, list_benchmarks
from cbf.providers import ProviderRegistry, build_registry_from_settings, parse_model_spec
from cbf.storage import BudgetGuard, CallLogger


def expand_conditions(conditions: dict) -> dict:
    """Normalize a conditions spec into a mapping of cell name -> run config.

    Accepts either an explicit ``{name: config}`` mapping (returned unchanged) or a
    factorial grid ``{factor: [levels...], "samples_per_cell": N}`` expanded into cells.
    """
    if not conditions:
        return conditions
    if all(isinstance(v, dict) for v in conditions.values()):
        return conditions

    list_factors = {k: v for k, v in conditions.items() if isinstance(v, list)}
    if not list_factors:
        return conditions

    scalars = {k: v for k, v in conditions.items() if not isinstance(v, list)}
    if "samples_per_cell" in scalars:
        scalars["num_samples"] = scalars.pop("samples_per_cell")

    cells: dict[str, dict] = {}
    keys = list(list_factors.keys())
    for combo in product(*(list_factors[k] for k in keys)):
        factor = dict(zip(keys, combo, strict=False))
        framing = str(factor.get("prompt_framing", "neutral")).lower()
        cell_cfg = {"condition": "creative" if "creat" in framing else "neutral", **scalars, **factor}
        name = ",".join(f"{k}={factor[k]}" for k in keys)
        cells[name] = cell_cfg
    return cells


class _LoggingProvider:
    """Wraps a provider so every completion is persisted to the run call log."""

    def __init__(self, provider, logger: CallLogger, budget: BudgetGuard | None):
        self._provider = provider
        self._logger = logger
        self._budget = budget
        self.name = getattr(provider, "name", "unknown")

    def __getattr__(self, item):
        return getattr(self._provider, item)

    async def complete(self, req):
        resp = await self._provider.complete(req)
        try:
            cost = float(self._provider.cost_usd(resp))
        except Exception:
            cost = 0.0
        if self._budget is not None:
            self._budget.check_and_add(cost)
        self._logger.log_call(
            provider=self.name,
            model=req.model,
            request=asdict(req),
            response={"text": resp.text, "model": resp.model, "raw": resp.raw},
            latency_s=resp.latency_s,
            cost_usd=cost,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )
        return resp


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
        self.run_id = f"{model_spec.replace(':', '-')}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
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

    def _resolve_logged_provider(self, model_spec: str):
        """Resolve a judge model spec to a provider, wrapped so judge calls are logged."""
        provider_name, _ = parse_model_spec(model_spec)
        provider = self.registry.get(provider_name) if self.registry else None
        if not provider:
            raise ValueError(f"Judge provider {provider_name} not configured")
        if self.logger is None:
            return provider
        return _LoggingProvider(provider, self.logger, self.budget)

    async def run_benchmark(self, provider, model: str, benchmark_name: str, conditions: dict) -> dict:
        bench = get_benchmark(benchmark_name)
        bench_config = self.config.get("benchmarks", {}).get(benchmark_name, {})

        if self.logger is not None and not isinstance(provider, _LoggingProvider):
            provider = _LoggingProvider(provider, self.logger, self.budget)

        all_results = {}

        for condition_name, condition_cfg in conditions.items():
            run_cfg = {**bench_config, **condition_cfg}
            run_cfg.setdefault("condition", condition_name)
            if self.config.get("judges"):
                run_cfg.setdefault("judges", self.config.get("judges"))

            raw_results = await bench.run(provider, model, run_cfg)

            judge_fn = getattr(bench, "judge", None)
            if callable(judge_fn):
                await judge_fn(raw_results, run_cfg, self._resolve_logged_provider)

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
        run_id = f"{model_spec.replace(':', '-')}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        runner.run_id = run_id
        runner.logger = CallLogger(run_id, runs_dir=str(runs_dir))
        runner.logger.update_manifest(
            models=[model_spec],
            config_hash=runner._hash_config(),
        )
        runner.finalize(status="dry_run")
        return {"run_id": run_id, "status": "dry_run", "estimated_cost": 0.0}

    provider, model = await runner.initialize(model_spec, max_usd)
    if runner.logger is not None:
        runner.logger.update_manifest(suite=suite)

    if conditions is None:
        conditions = config.get("conditions", {})
        if not conditions:
            conditions = {
                "neutral": {"condition": "neutral", "temperature": 0.7, "num_samples": 10},
                "creative": {"condition": "creative", "temperature": 0.7, "num_samples": 10},
            }
    conditions = expand_conditions(conditions)

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
