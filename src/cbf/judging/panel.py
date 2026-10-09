import asyncio
import random
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np
from scipy import stats
import krippendorff

from cbf.providers import LLMProvider, LLMRequest
from cbf.benchmarks import Benchmark, BenchmarkItem, RawResults, Scores


@dataclass
class JudgeConfig:
    judge_models: list[str] = field(default_factory=lambda: [
        "openai:gpt-4o",
        "anthropic:claude-3-5-sonnet-20241022",
    ])
    temperature: float = 0.0
    max_tokens: int = 1000
    rubric: str = ""
    swap_position: bool = True


@dataclass
class PairwiseResult:
    item_a_id: str
    item_b_id: str
    winner: str  # "a", "b", or "tie"
    judge: str
    position_swapped: bool
    reasoning: str = ""
    raw_response: str = ""


class JudgePanel:
    def __init__(self, config: JudgeConfig, provider_registry):
        self.config = config
        self.provider_registry = provider_registry
        self.judges: dict[str, LLMProvider] = {}

    async def initialize(self):
        for model_spec in self.config.judge_models:
            provider_name, model_name = model_spec.split(":", 1)
            provider = self.provider_registry.get(provider_name)
            if not provider:
                raise ValueError(f"Judge provider {provider_name} not configured")
            self.judges[model_spec] = provider

    JUDGE_PROMPT = """You are an expert evaluator of creative writing. Compare two texts and decide which is more creative, original, and well-written.

Rubric:
{rubric}

Text A:
{text_a}

Text B:
{text_b}

Which text is better? Respond with ONLY one of: "A", "B", or "TIE".
Then provide a brief reasoning (1-2 sentences)."""

    async def judge_pair(
        self,
        judge_model: str,
        text_a: str,
        text_b: str,
        item_a_id: str,
        item_b_id: str,
        swap: bool = False,
    ) -> PairwiseResult:
        provider = self.judges[judge_model]

        if swap:
            prompt = self.JUDGE_PROMPT.format(
                rubric=self.config.rubric,
                text_a=text_b,
                text_b=text_a,
            )
        else:
            prompt = self.JUDGE_PROMPT.format(
                rubric=self.config.rubric,
                text_a=text_a,
                text_b=text_b,
            )

        req = LLMRequest(
            model=judge_model.split(":", 1)[1],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        resp = await provider.complete(req)
        response_text = resp.text.strip().upper()

        winner = "tie"
        if response_text.startswith("A"):
            winner = "b" if swap else "a"
        elif response_text.startswith("B"):
            winner = "a" if swap else "b"

        return PairwiseResult(
            item_a_id=item_a_id,
            item_b_id=item_b_id,
            winner=winner,
            judge=judge_model,
            position_swapped=swap,
            reasoning=response_text,
            raw_response=resp.text,
        )

    async def judge_all_pairs(
        self,
        texts: dict[str, str],
        item_ids: list[str] | None = None,
    ) -> list[PairwiseResult]:
        if item_ids is None:
            item_ids = list(texts.keys())

        results = []
        pairs = []
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                pairs.append((item_ids[i], item_ids[j]))

        for judge_model in self.config.judge_models:
            for item_a_id, item_b_id in pairs:
                # First judgment (original position)
                result1 = await self.judge_pair(
                    judge_model,
                    texts[item_a_id],
                    texts[item_b_id],
                    item_a_id,
                    item_b_id,
                    swap=False,
                )
                results.append(result1)

                # Second judgment (swapped position) if enabled
                if self.config.swap_position:
                    result2 = await self.judge_pair(
                        judge_model,
                        texts[item_a_id],
                        texts[item_b_id],
                        item_a_id,
                        item_b_id,
                        swap=True,
                    )
                    results.append(result2)

        return results


def compute_elo(
    results: list[PairwiseResult],
    k_factor: float = 32.0,
    initial_rating: float = 1000.0,
) -> dict[str, float]:
    items = set()
    for r in results:
        items.add(r.item_a_id)
        items.add(r.item_b_id)

    ratings = {item: initial_rating for item in items}

    for result in results:
        if result.winner == "tie":
            continue

        winner_id = result.item_a_id if result.winner == "a" else result.item_b_id
        loser_id = result.item_b_id if result.winner == "a" else result.item_a_id

        r_w = ratings[winner_id]
        r_l = ratings[loser_id]

        e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        e_l = 1 / (1 + 10 ** ((r_w - r_l) / 400))

        ratings[winner_id] += k_factor * (1 - e_w)
        ratings[loser_id] += k_factor * (0 - e_l)

    return ratings


def compute_bradley_terry(
    results: list[PairwiseResult],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[str, float]:
    items = set()
    for r in results:
        items.add(r.item_a_id)
        items.add(r.item_b_id)

    item_list = sorted(items)
    n = len(item_list)
    item_to_idx = {item: i for i, item in enumerate(item_list)}

    wins = np.zeros((n, n))
    for r in results:
        if r.winner == "tie":
            continue
        i = item_to_idx[r.item_a_id if r.winner == "a" else r.item_b_id]
        j = item_to_idx[r.item_b_id if r.winner == "a" else r.item_a_id]
        wins[i, j] += 1

    params = np.ones(n)
    for _ in range(max_iter):
        new_params = np.zeros(n)
        for i in range(n):
            numerator = np.sum(wins[i, :])
            denominator = 0
            for j in range(n):
                if i != j:
                    denominator += (wins[i, j] + wins[j, i]) / (params[i] + params[j])
            if denominator > 0:
                new_params[i] = numerator / denominator
            else:
                new_params[i] = params[i]

        if np.max(np.abs(new_params - params)) < tol:
            params = new_params
            break
        params = new_params

    params = params / np.sum(params) * 1000
    return {item_list[i]: float(params[i]) for i in range(n)}


def compute_inter_judge_agreement(results: list[PairwiseResult]) -> dict:
    judges = list(set(r.judge for r in results))
    items = set()
    for r in results:
        items.add(r.item_a_id)
        items.add(r.item_b_id)
    item_list = sorted(items)

    judge_ratings = {}
    for judge in judges:
        judge_results = [r for r in results if r.judge == judge]
        ratings = compute_elo(judge_results)
        judge_ratings[judge] = [ratings.get(item, 1000) for item in item_list]

    matrix = np.array([judge_ratings[j] for j in judges])
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")

    spearman_corrs = []
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            corr, _ = stats.spearmanr(judge_ratings[judges[i]], judge_ratings[judges[j]])
            if not np.isnan(corr):
                spearman_corrs.append(corr)

    return {
        "krippendorff_alpha": float(alpha) if not np.isnan(alpha) else 0.0,
        "mean_spearman": float(np.mean(spearman_corrs)) if spearman_corrs else 0.0,
        "num_judges": len(judges),
        "num_items": len(item_list),
    }


class WritingEloBenchmark:
    id = "writing_elo"
    version = "1.0.0"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.prompts_list = self.config.get("prompts", [
            "Write a short story about a robot learning to paint.",
            "Describe a world where gravity works sideways.",
            "Invent a new sport played on the moon.",
            "Write a poem about a color that doesn't exist.",
            "Describe a conversation between a cloud and a mountain.",
        ])
        self.rubric = self.config.get("rubric", """
- Originality: Does the piece present novel ideas, concepts, or perspectives?
- Creativity: Does it show imaginative thinking and creative problem-solving?
- Writing Quality: Is the prose well-crafted, engaging, and coherent?
- Emotional Resonance: Does it evoke genuine emotion or insight?
""")

    def prompts(self) -> list[BenchmarkItem]:
        return [
            BenchmarkItem(
                id=f"writing_{i}",
                prompt=prompt,
                metadata={"prompt_idx": i},
            )
            for i, prompt in enumerate(self.prompts_list)
        ]

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        k = cfg.get("k_generations", 1)
        temperature = cfg.get("temperature", 0.8)

        all_items = []
        call_ids = []

        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))

        async def run_generation(prompt_item: BenchmarkItem, gen_idx: int):
            async with semaphore:
                req = LLMRequest(
                    model=model,
                    messages=[{"role": "user", "content": prompt_item.prompt}],
                    temperature=temperature,
                    max_tokens=500,
                    seed=cfg.get("seed"),
                )
                resp = await provider.complete(req)
                call_ids.append(resp.request_id)
                return {
                    "prompt_id": prompt_item.id,
                    "prompt_idx": prompt_item.metadata["prompt_idx"],
                    "generation_idx": gen_idx,
                    "text": resp.text,
                    "call_id": resp.request_id,
                    "latency_s": resp.latency_s,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }

        tasks = []
        for prompt_item in self.prompts():
            for i in range(k):
                tasks.append(run_generation(prompt_item, i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                all_items.append({"prompt_id": "unknown", "error": str(r), "text": ""})
            else:
                all_items.append(r)

        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=all_items,
            call_ids=call_ids,
            metadata={"config": cfg, "k": k},
        )

    def score(self, raw: RawResults) -> Scores:
        by_prompt: dict[int, list[dict]] = {}
        for item in raw.items:
            if "error" in item:
                continue
            idx = item.get("prompt_idx", 0)
            if idx not in by_prompt:
                by_prompt[idx] = []
            by_prompt[idx].append(item)

        if not by_prompt:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"elo_rating": 0.0, "agreement": 0.0},
            )

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={"items_generated": len(raw.items), "prompts_tested": len(by_prompt)},
            metadata={"by_prompt": {str(k): len(v) for k, v in by_prompt.items()}},
        )


async def run_writing_elo_with_judges(
    provider_registry,
    model: str,
    prompts: list[str],
    judge_models: list[str],
    k_generations: int = 1,
    temperature: float = 0.8,
    rubric: str = "",
) -> dict:
    bench = WritingEloBenchmark({
        "prompts": prompts,
        "rubric": rubric,
    })

    judge_config = JudgeConfig(
        judge_models=judge_models,
        rubric=rubric,
    )
    panel = JudgePanel(judge_config, provider_registry)
    await panel.initialize()

    all_texts = {}
    item_ids = []

    for prompt_idx, prompt in enumerate(prompts):
        for gen_idx in range(k_generations):
            item_id = f"p{prompt_idx}_g{gen_idx}"
            item_ids.append(item_id)

            req = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            resp = await provider_registry.get(model.split(":")[0]).complete(req)
            all_texts[item_id] = resp.text

    results = await panel.judge_all_pairs(all_texts, item_ids)

    elo_ratings = compute_elo(results)
    bt_ratings = compute_bradley_terry(results)
    agreement = compute_inter_judge_agreement(results)

    return {
        "texts": all_texts,
        "pairwise_results": results,
        "elo_ratings": elo_ratings,
        "bt_ratings": bt_ratings,
        "agreement": agreement,
    }