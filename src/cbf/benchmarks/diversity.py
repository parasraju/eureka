import asyncio
import random
import re
from dataclasses import dataclass, field

import numpy as np

from cbf.judging.panel import build_judge_config, is_self_judge
from cbf.metrics import (
    cluster_by_similarity,
    distinct_n,
    mean_ci,
    mean_pairwise_distance,
    self_bleu,
)

from .base import BenchmarkItem, RawResults, Scores

_PRIVATE_PROMPT_TEMPLATES = [
    "Write a short story in which a {a} encounters a {b}.",
    "Describe a world where {a} and {b} follow incompatible rules.",
    "Invent a ritual that requires both a {a} and a {b}.",
    "Write a poem about the distance between a {a} and a {b}.",
    "Describe a conversation between a {a} and a {b}.",
    "Design a machine whose only inputs are a {a} and a {b}.",
    "Write a letter from a {a} to a {b} that arrives a century late.",
    "Describe an unexpected friendship between a {a} and a {b}.",
]

_PRIVATE_NOUNS = [
    "anchor", "accordion", "bakery", "bicycle", "bridge", "cactus", "camera",
    "candle", "castle", "compass", "desert", "dictionary", "engine", "forest",
    "garden", "glacier", "harbor", "hospital", "island", "keyboard", "lantern",
    "library", "lighthouse", "magnet", "mirror", "museum", "observatory",
    "orchard", "parachute", "piano", "railway", "river", "satellite", "school",
    "sculpture", "submarine", "telescope", "theater", "tunnel", "volcano",
]


@dataclass
class DiversityConfig:
    prompts: list[str] = field(default_factory=lambda: [
        "Write a short story about a robot learning to paint.",
        "Describe a world where gravity works sideways.",
        "Invent a new sport played on the moon.",
        "Write a poem about a color that doesn't exist.",
        "Describe a conversation between a cloud and a mountain.",
    ])
    k_generations: int = 10
    temperature: float = 0.8
    embedding_model: str = "sentence_transformers"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75
    quality_judge_prompt: str = "Rate the creativity and quality of this text from 1-10. Respond with only the number."

    @classmethod
    def private_prompt_set(cls, seed: int, n: int) -> list[str]:
        """Deterministically build fresh prompts from templates and seeded noun pairs."""
        if n <= 0:
            return []
        rng = random.Random(seed)
        prompts: list[str] = []
        seen: set[str] = set()
        attempts = 0
        while len(prompts) < n and attempts < max(100, n * 50):
            attempts += 1
            a, b = rng.sample(_PRIVATE_NOUNS, 2)
            template = _PRIVATE_PROMPT_TEMPLATES[len(prompts) % len(_PRIVATE_PROMPT_TEMPLATES)]
            prompt = template.format(a=a, b=b)
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
        return prompts


def _parse_rating(text: str) -> float | None:
    """Extract a 1-10 rating from a judge response, or None if unparseable."""
    if not text:
        return None
    match = re.search(r"\b(10|[1-9])\b", text)
    if not match:
        return None
    return float(match.group(1))


class DiversityBenchmark:
    id = "diversity"
    version = "1.0.0"

    def __init__(self, config: DiversityConfig | None = None):
        self.config = config or DiversityConfig()
        self._st_model = None

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.config.st_model)
        return self._st_model

    def prompts(self) -> list[BenchmarkItem]:
        return [
            BenchmarkItem(
                id=f"diversity_{i}",
                prompt=prompt,
                metadata={"prompt_idx": i},
            )
            for i, prompt in enumerate(self.config.prompts)
        ]

    def _run_prompts(self, cfg: dict) -> tuple[list[BenchmarkItem], dict]:
        if not cfg.get("private_prompts"):
            return self.prompts(), {}

        seed = cfg.get("seed")
        seed = 0 if seed is None else int(seed)
        n = cfg.get("private_prompt_count", len(self.config.prompts))
        private_value = cfg.get("private_prompts")
        if isinstance(private_value, int) and not isinstance(private_value, bool) and private_value > 0:
            n = private_value
        texts = self.config.private_prompt_set(seed, int(n))
        items = [
            BenchmarkItem(
                id=f"diversity_private_{i}",
                prompt=prompt,
                metadata={"prompt_idx": i, "provenance": "private-seeded"},
            )
            for i, prompt in enumerate(texts)
        ]
        return items, {"prompt_provenance": "private-seeded", "seed": seed}

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        from cbf.providers import LLMRequest

        k = cfg.get("k_generations", self.config.k_generations)
        temperature = cfg.get("temperature", self.config.temperature)
        prompt_items, provenance = self._run_prompts(cfg)

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
        for prompt_item in prompt_items:
            for i in range(k):
                tasks.append(run_generation(prompt_item, i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                all_items.append({"prompt_id": "unknown", "error": str(r), "text": ""})
            else:
                all_items.append(r)

        metadata = {"config": cfg, "k": k, **provenance}
        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=all_items,
            call_ids=call_ids,
            metadata=metadata,
        )

    async def judge(self, raw: RawResults, cfg: dict, resolve_provider) -> None:
        """Score each generation 1-10 with every independent judge in the panel."""
        judge_config = build_judge_config(cfg.get("judges"))
        if not judge_config.judge_models:
            return

        selected = [s for s in judge_config.judge_models if not is_self_judge(s, raw.model)]
        filtered = [s for s in judge_config.judge_models if is_self_judge(s, raw.model)]
        if not selected:
            raise ValueError(
                "All configured judges are the model under test (self-preference); "
                "configure at least one independent judge."
            )
        if resolve_provider is None:
            raise ValueError("Judging requires a provider resolver to log judge calls")

        providers = {spec: resolve_provider(spec) for spec in selected}
        rubric = cfg.get("quality_judge_prompt", self.config.quality_judge_prompt)

        from cbf.providers import LLMRequest

        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))
        rated_items = [
            (f"p{item.get('prompt_idx', 0)}_g{item.get('generation_idx', 0)}", item.get("text", ""))
            for item in raw.items
            if "error" not in item and item.get("text")
        ]

        async def rate(judge_spec: str, item_id: str, text: str) -> tuple[str, float | None]:
            async with semaphore:
                provider = providers[judge_spec]
                req = LLMRequest(
                    model=judge_spec.split(":", 1)[1],
                    messages=[{"role": "user", "content": f"{rubric}\n\nText:\n{text}"}],
                    temperature=judge_config.temperature,
                    max_tokens=judge_config.max_tokens,
                )
                resp = await provider.complete(req)
                return item_id, _parse_rating(resp.text)

        results = await asyncio.gather(
            *(rate(spec, item_id, text) for spec in selected for item_id, text in rated_items),
            return_exceptions=True,
        )

        per_item: dict[str, list[float]] = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            item_id, rating = result
            if rating is not None:
                per_item.setdefault(item_id, []).append(rating)

        raw.metadata["quality"] = {
            item_id: float(np.mean(ratings)) for item_id, ratings in per_item.items()
        }
        raw.metadata["judge"] = {
            "models": selected,
            "filtered_self_judges": filtered,
            "metric": "quality",
        }

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        model = self._get_st_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _compute_distinct_n(self, texts: list[str], n: int) -> float:
        return distinct_n(texts, n)

    def _compute_self_bleu(self, texts: list[str]) -> float:
        return self_bleu(texts)

    def _cluster_by_similarity(self, embeddings: np.ndarray, threshold: float) -> list[list[int]]:
        return cluster_by_similarity(embeddings, threshold)

    def score(self, raw: RawResults) -> Scores:
        by_prompt: dict[int, list[dict]] = {}
        for item in raw.items:
            if "error" in item:
                continue
            idx = item.get("prompt_idx", 0)
            if idx not in by_prompt:
                by_prompt[idx] = []
            by_prompt[idx].append(item)

        quality_by_item = raw.metadata.get("quality") or {}
        if not isinstance(quality_by_item, dict):
            quality_by_item = {}

        if not by_prompt:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"diversity_score": 0.0},
                metadata={"composite_available": False},
            )

        all_diversity = []
        all_quality = []
        all_ratings = []
        all_distinct_1 = []
        all_distinct_2 = []
        all_self_bleu = []
        all_num_clusters = []
        all_generation_counts = []
        per_prompt_stats: dict[int, dict] = {}
        composites: list[float] = []

        for prompt_idx, items in by_prompt.items():
            texts = [item["text"] for item in items if item.get("text")]
            all_generation_counts.append(len(texts))
            per_prompt_stats[prompt_idx] = {"generations": len(texts)}

            prompt_ratings = []
            for item in items:
                item_id = f"p{prompt_idx}_g{item.get('generation_idx', 0)}"
                if item_id in quality_by_item:
                    rating = float(quality_by_item[item_id])
                    prompt_ratings.append(rating)
                    all_ratings.append(rating)
            prompt_quality = float(np.mean(prompt_ratings)) / 10.0 if prompt_ratings else None
            if prompt_quality is not None:
                all_quality.append(prompt_quality)
                per_prompt_stats[prompt_idx]["quality"] = prompt_quality

            if len(texts) < 2:
                # A prompt with failed/insufficient generations contributes zero,
                # so flaky models are penalized instead of quietly dropped.
                all_diversity.append(0.0)
                all_num_clusters.append(0.0)
                all_distinct_1.append(0.0)
                all_distinct_2.append(0.0)
                all_self_bleu.append(0.0)
                per_prompt_stats[prompt_idx].update(
                    {"diversity": 0.0, "clusters": 0, "insufficient_generations": True}
                )
                if prompt_quality is not None:
                    composites.append(0.0)
                continue

            embeddings = self._get_embeddings(texts)

            mean_dist = mean_pairwise_distance(embeddings)
            all_diversity.append(mean_dist)

            clusters = self._cluster_by_similarity(embeddings, self.config.similarity_threshold)
            num_clusters = len(clusters)
            all_num_clusters.append(num_clusters)

            distinct_1 = self._compute_distinct_n(texts, 1)
            distinct_2 = self._compute_distinct_n(texts, 2)
            all_distinct_1.append(distinct_1)
            all_distinct_2.append(distinct_2)

            self_bleu_score = self._compute_self_bleu(texts)
            all_self_bleu.append(self_bleu_score)

            per_prompt_stats[prompt_idx].update({
                "diversity": mean_dist,
                "clusters": num_clusters,
            })
            if prompt_quality is not None:
                composites.append(mean_dist * prompt_quality)

        expected_prompts = max(by_prompt) + 1 if by_prompt else 0
        for _ in range(max(0, expected_prompts - len(by_prompt))):
            all_diversity.append(0.0)
            all_generation_counts.append(0)
            all_num_clusters.append(0.0)
            all_distinct_1.append(0.0)
            all_distinct_2.append(0.0)
            all_self_bleu.append(0.0)

        total_texts = int(sum(all_generation_counts))
        generation_failure_rate = (
            1.0 - total_texts / len(raw.items) if raw.items else 0.0
        )
        diversity_mean = float(np.mean(all_diversity)) if all_diversity else 0.0
        metrics: dict[str, float] = {
            "diversity_score": diversity_mean,
            "distinct_1": float(np.mean(all_distinct_1)) if all_distinct_1 else 0.0,
            "distinct_2": float(np.mean(all_distinct_2)) if all_distinct_2 else 0.0,
            "self_bleu": float(np.mean(all_self_bleu)) if all_self_bleu else 0.0,
            "num_clusters": float(np.mean(all_num_clusters)) if all_num_clusters else 0.0,
            "prompts_tested": len(by_prompt),
            "generations_per_prompt": (
                float(np.mean(all_generation_counts)) if all_generation_counts else 0.0
            ),
            "generation_failure_rate": float(generation_failure_rate),
        }
        confidence_intervals = {"diversity_score": mean_ci(all_diversity)}

        if all_ratings:
            metrics["quality_score"] = float(np.mean(all_ratings)) / 10.0
        if all_quality:
            confidence_intervals["quality_score"] = mean_ci(all_quality)

        composite_available = bool(composites)
        if composite_available:
            metrics["composite"] = float(np.mean(composites))
            confidence_intervals["composite"] = mean_ci(composites)

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics=metrics,
            confidence_intervals=confidence_intervals,
            metadata={
                "per_prompt": {str(idx): stats for idx, stats in per_prompt_stats.items()},
                "composite_available": composite_available,
            },
        )


def create_diversity_benchmark(config: dict | None = None) -> DiversityBenchmark:
    if config:
        div_config = DiversityConfig(**config)
    else:
        div_config = DiversityConfig()
    return DiversityBenchmark(div_config)
