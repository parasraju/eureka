import asyncio
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sacrebleu.metrics import BLEU

from .base import Benchmark, BenchmarkItem, RawResults, Scores


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

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        from cbf.providers import LLMRequest

        k = cfg.get("k_generations", self.config.k_generations)
        temperature = cfg.get("temperature", self.config.temperature)

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

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        model = self._get_st_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _compute_distinct_n(self, texts: list[str], n: int) -> float:
        all_ngrams = []
        unique_ngrams = set()
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                all_ngrams.append(ngram)
                unique_ngrams.add(ngram)
        return len(unique_ngrams) / len(all_ngrams) if all_ngrams else 0.0

    def _compute_self_bleu(self, texts: list[str]) -> float:
        if len(texts) < 2:
            return 0.0
        bleu = BLEU()
        scores = []
        for i, text in enumerate(texts):
            refs = [t for j, t in enumerate(texts) if j != i]
            score = bleu.sentence_score(text, refs).score / 100.0
            scores.append(score)
        return float(np.mean(scores))

    def _cluster_by_similarity(self, embeddings: np.ndarray, threshold: float) -> list[list[int]]:
        n = len(embeddings)
        if n == 0:
            return []
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j]:
                    sim = np.dot(embeddings[i], embeddings[j])
                    if sim >= threshold:
                        cluster.append(j)
                        visited[j] = True
            clusters.append(cluster)

        return clusters

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
                metrics={"diversity_score": 0.0, "quality_score": 0.0, "composite": 0.0},
            )

        all_diversity = []
        all_quality = []
        all_distinct_1 = []
        all_distinct_2 = []
        all_self_bleu = []
        all_num_clusters = []

        for prompt_idx, items in by_prompt.items():
            texts = [item["text"] for item in items if item.get("text")]
            if len(texts) < 2:
                continue

            embeddings = self._get_embeddings(texts)

            distances = cosine_distances(embeddings)
            np.fill_diagonal(distances, np.nan)
            mean_dist = float(np.nanmean(distances))
            all_diversity.append(mean_dist)

            clusters = self._cluster_by_similarity(embeddings, self.config.similarity_threshold)
            num_clusters = len(clusters)
            all_num_clusters.append(num_clusters)

            distinct_1 = self._compute_distinct_n(texts, 1)
            distinct_2 = self._compute_distinct_n(texts, 2)
            all_distinct_1.append(distinct_1)
            all_distinct_2.append(distinct_2)

            self_bleu = self._compute_self_bleu(texts)
            all_self_bleu.append(self_bleu)

            quality = num_clusters * mean_dist
            all_quality.append(quality)

        from scipy import stats

        def ci(data):
            if len(data) < 2:
                return (float(np.mean(data)) if data else 0.0, float(np.mean(data)) if data else 0.0)
            mean = float(np.mean(data))
            sem = stats.sem(data)
            low, high = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
            return (float(low), float(high))

        diversity_mean = float(np.mean(all_diversity)) if all_diversity else 0.0
        quality_mean = float(np.mean(all_quality)) if all_quality else 0.0
        composite = diversity_mean * quality_mean if quality_mean > 0 else 0.0

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "diversity_score": diversity_mean,
                "quality_score": quality_mean,
                "composite": composite,
                "distinct_1": float(np.mean(all_distinct_1)) if all_distinct_1 else 0.0,
                "distinct_2": float(np.mean(all_distinct_2)) if all_distinct_2 else 0.0,
                "self_bleu": float(np.mean(all_self_bleu)) if all_self_bleu else 0.0,
                "num_clusters": float(np.mean(all_num_clusters)) if all_num_clusters else 0.0,
                "prompts_tested": len(by_prompt),
                "generations_per_prompt": self.config.k_generations,
            },
            confidence_intervals={
                "diversity_score": ci(all_diversity),
                "quality_score": ci(all_quality),
                "composite": ci([d * q for d, q in zip(all_diversity, all_quality)]) if all_diversity and all_quality else (0.0, 0.0),
            },
            metadata={
                "per_prompt": {
                    str(idx): {
                        "diversity": all_diversity[i],
                        "quality": all_quality[i],
                        "clusters": all_num_clusters[i],
                    }
                    for i, idx in enumerate(by_prompt.keys())
                }
            },
        )


def create_diversity_benchmark(config: dict | None = None) -> DiversityBenchmark:
    if config:
        div_config = DiversityConfig(**config)
    else:
        div_config = DiversityConfig()
    return DiversityBenchmark(div_config)