"""Dry-run cost estimation for benchmark suites.

The estimator mirrors each benchmark's ``run`` semantics:

* DAT: one call per sample (only one prompt variant is used per condition)
* AUT: ``len(objects) * samples`` calls
* diversity: ``len(prompts) * k_generations`` calls
* associative: ``len(pairs) * samples`` calls
* affordance: ``len(tasks) * samples`` calls
* writing_elo: ``len(prompts) * k_generations`` calls (judge calls excluded)

Counts are summed across every expanded condition cell. Token counts are
approximate (4 characters per prompt token) and prices come from a small,
explicit table. Unknown models use the documented default price. Results are
rounded to cents; the ``assumptions`` list records every approximation.
"""

import math
from typing import Any

from cbf.benchmarks import TIER_A, TIER_B, TIER_C, get_benchmark, list_benchmarks

# USD per 1M tokens: model -> (input, output)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "gemini-1.5-pro": (1.25, 5.00),
}
DEFAULT_PRICE = (5.00, 15.00)  # USD per 1M tokens for unlisted models

MAX_OUTPUT_TOKENS = {
    "dat": 100,
    "aut": 300,
    "diversity": 500,
    "associative": 120,
    "affordance": 400,
    "writing_elo": 500,
}

CHARS_PER_TOKEN = 4


def _price_for_model(model_spec: str) -> tuple[float, float]:
    model = model_spec.split(":", 1)[1] if ":" in model_spec else model_spec
    lowered = model.lower()
    if lowered in MODEL_PRICES:
        return MODEL_PRICES[lowered]
    for known, price in MODEL_PRICES.items():
        if lowered.startswith(known):
            return price
    return DEFAULT_PRICE


def _token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN)) if text else 0


def _resolve_suite(suite: str) -> list[str]:
    names: list[str] = []
    for raw in suite.split(","):
        part = raw.strip().lower()
        if not part:
            continue
        if part == "tier_a":
            names.extend(TIER_A)
        elif part == "tier_b":
            names.extend(TIER_B)
        elif part == "tier_c":
            names.extend(TIER_C)
        elif part in ("all", "*"):
            names.extend(list_benchmarks())
        elif part in list_benchmarks():
            names.append(part)
        else:
            raise ValueError(f"Unknown suite or benchmark: {part!r}")
    return list(dict.fromkeys(names))


def _benchmark_config(config: dict, name: str) -> dict:
    benchmarks = config.get("benchmarks", {}) if isinstance(config, dict) else {}
    value = benchmarks.get(name, {}) if isinstance(benchmarks, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _cell_usage(benchmark_name: str, bench: Any, cell: dict) -> tuple[int, int]:
    """Return (calls, input_tokens) for one benchmark in one condition cell."""
    items = bench.prompts()
    prompt_tokens = [_token_count(item.prompt) for item in items]

    if benchmark_name == "dat":
        samples = int(cell.get("num_samples", bench.config.num_samples))
        calls = samples
        if prompt_tokens:
            average = max(1, sum(prompt_tokens) // len(prompt_tokens))
            input_tokens = samples * average
        else:
            input_tokens = 0
        return calls, input_tokens

    if benchmark_name == "aut":
        samples = int(cell.get("num_samples", bench.config.num_samples))
        calls = len(bench.config.objects) * samples
        if items:
            average = max(1, sum(prompt_tokens) // len(prompt_tokens))
            input_tokens = calls * average
        else:
            input_tokens = 0
        return calls, input_tokens

    if benchmark_name == "diversity":
        k = int(cell.get("k_generations", bench.config.k_generations))
        return len(items) * k, sum(prompt_tokens) * k

    if benchmark_name == "associative":
        samples = int(cell.get("num_samples", bench.config.num_samples))
        calls = len(bench.config.pairs) * samples
        return calls, sum(prompt_tokens) * samples

    if benchmark_name == "affordance":
        samples = int(cell.get("num_samples", bench.config.num_samples))
        calls = len(bench.config.tasks) * samples
        return calls, sum(prompt_tokens) * samples

    if benchmark_name == "writing_elo":
        k = int(cell.get("k_generations", 1))
        return len(items) * k, sum(prompt_tokens) * k

    raise ValueError(f"Unsupported benchmark for costing: {benchmark_name!r}")


def _default_conditions(config: dict) -> dict:
    configured = config.get("conditions", {}) if isinstance(config, dict) else {}
    cells = {}
    if configured:
        from cbf.runner import expand_conditions

        cells = expand_conditions(configured)
    if cells:
        return cells
    return {
        "neutral": {"condition": "neutral", "temperature": 0.7, "num_samples": 10},
        "creative": {"condition": "creative", "temperature": 0.7, "num_samples": 10},
    }


def estimate_run_cost(
    config: dict,
    suite: str,
    conditions: dict,
    model_spec: str,
) -> dict:
    """Estimate calls, tokens, and USD for a run without making API calls."""
    from cbf.runner import expand_conditions

    names = _resolve_suite(suite)
    cells = expand_conditions(conditions) if conditions else _default_conditions(config)
    if not cells:
        cells = _default_conditions(config)

    input_price, output_price = _price_for_model(model_spec)
    breakdown: dict[str, dict] = {}
    total_calls = 0
    total_input = 0
    total_output = 0
    total_cost = 0.0

    for name in sorted(names):
        bench_config = _benchmark_config(config, name)
        try:
            bench = get_benchmark(name, bench_config or None)
        except TypeError:
            bench = get_benchmark(name)

        calls = 0
        input_tokens = 0
        for cell in cells.values():
            cell_calls, cell_input = _cell_usage(name, bench, cell)
            calls += cell_calls
            input_tokens += cell_input

        output_tokens = calls * MAX_OUTPUT_TOKENS[name]
        cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

        breakdown[name] = {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_usd": round(cost, 2),
        }
        total_calls += calls
        total_input += input_tokens
        total_output += output_tokens
        total_cost += cost

    assumptions = [
        f"Token counts use {CHARS_PER_TOKEN} characters per token (approximate).",
        f"Prices for {model_spec!r}: ${input_price:.2f}/1M input, ${output_price:.2f}/1M output.",
        "Unlisted models use the default price of $5.00/1M input and $15.00/1M output.",
        "Output tokens equal each benchmark's max_tokens; models may use fewer.",
        "Condition cells are expanded with cbf.runner.expand_conditions; per-cell overrides are honored.",
        "writing_elo judge-panel calls are not included in this estimate.",
        "Totals are rounded to cents; per-benchmark costs may not sum exactly to the total.",
    ]

    return {
        "calls": total_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_usd": round(total_cost, 2),
        "breakdown": breakdown,
        "assumptions": assumptions,
    }
