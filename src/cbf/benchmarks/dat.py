import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from .base import BenchmarkItem, RawResults, Scores


@dataclass
class DATConfig:
    num_words: int = 10
    num_samples: int = 50
    min_word_length: int = 3
    max_word_length: int = 12
    temperature: float = 0.7
    embedding_model: str = "glove"
    glove_path: str | None = None


class DATBenchmark:
    id = "dat"
    version = "1.0.0"

    DAT_PROMPT_NEUTRAL = """Please list {num_words} words that are as different from each other as possible, in terms of their meanings.
The words should be common English nouns (single words only, no phrases).
Format your response as a simple comma-separated list.

Example format: word1, word2, word3, word4, word5, word6, word7, word8, word9, word10"""

    DAT_PROMPT_CREATIVE = """Be as creative and original as possible. List {num_words} words that are as different from each other as possible, in terms of their meanings.
The words should be common English nouns (single words only, no phrases).
Format your response as a simple comma-separated list.

Example format: word1, word2, word3, word4, word5, word6, word7, word8, word9, word10"""

    def __init__(self, config: DATConfig | None = None):
        self.config = config or DATConfig()
        self._glove_vectors: dict[str, np.ndarray] | None = None
        self._glove_dim: int = 300

    def prompts(self) -> list[BenchmarkItem]:
        return [
            BenchmarkItem(
                id="dat_neutral",
                prompt=self.DAT_PROMPT_NEUTRAL.format(num_words=self.config.num_words),
                metadata={"condition": "neutral"},
            ),
            BenchmarkItem(
                id="dat_creative",
                prompt=self.DAT_PROMPT_CREATIVE.format(num_words=self.config.num_words),
                metadata={"condition": "creative"},
            ),
        ]

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        from cbf.providers import LLMRequest

        num_samples = cfg.get("num_samples", self.config.num_samples)
        temperature = cfg.get("temperature", self.config.temperature)
        condition = cfg.get("condition", "neutral")

        prompt_template = self.DAT_PROMPT_CREATIVE if condition == "creative" else self.DAT_PROMPT_NEUTRAL
        prompt = prompt_template.format(num_words=self.config.num_words)

        items = []
        call_ids = []

        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))

        async def run_sample(sample_idx: int):
            async with semaphore:
                req = LLMRequest(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=100,
                    seed=cfg.get("seed"),
                )
                resp = await provider.complete(req)
                call_ids.append(resp.request_id)
                return {
                    "sample_idx": sample_idx,
                    "response": resp.text,
                    "call_id": resp.request_id,
                    "latency_s": resp.latency_s,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }

        tasks = [run_sample(i) for i in range(num_samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                items.append({"sample_idx": -1, "error": str(r), "words": []})
            else:
                words = self._parse_words(r["response"])
                r["words"] = words
                r["valid"] = len(words) == self.config.num_words
                items.append(r)

        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=items,
            call_ids=call_ids,
            metadata={"config": cfg, "condition": condition},
        )

    def _parse_words(self, text: str) -> list[str]:
        text = text.lower().strip()
        words = re.split(r'[,\n;]+', text)
        cleaned = []
        for w in words:
            w = w.strip()
            w = re.sub(r'[^a-z]', '', w)
            if w and self.config.min_word_length <= len(w) <= self.config.max_word_length:
                cleaned.append(w)
        return cleaned[:self.config.num_words]

    def _load_glove(self) -> dict[str, np.ndarray]:
        if self._glove_vectors is not None:
            return self._glove_vectors

        if self.config.glove_path and Path(self.config.glove_path).exists():
            path = self.config.glove_path
        else:
            cache_dir = Path.home() / ".cache" / "cbf" / "glove"
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / "glove.6B.300d.txt"

        if not path.exists():
            raise FileNotFoundError(
                f"GloVe vectors not found at {path}. Download from https://nlp.stanford.edu/projects/glove/ "
                f"or set DATConfig.glove_path"
            )

        vectors = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == self._glove_dim + 1:
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors[word] = vec

        self._glove_vectors = vectors
        return vectors

    def _get_word_vectors(self, words: list[str]) -> np.ndarray:
        if self.config.embedding_model == "glove":
            glove = self._load_glove()
            vecs = []
            for w in words:
                if w in glove:
                    vecs.append(glove[w])
                else:
                    vecs.append(np.zeros(self._glove_dim, dtype=np.float32))
            return np.array(vecs)
        else:
            raise NotImplementedError(f"Embedding model {self.config.embedding_model} not implemented")

    def score(self, raw: RawResults) -> Scores:
        valid_items = [item for item in raw.items if item.get("valid", False)]
        if not valid_items:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"dat_score": 0.0, "valid_samples": 0},
                metadata={"error": "No valid samples"},
            )

        all_scores = []
        for item in valid_items:
            words = item.get("words", [])
            if len(words) < 2:
                continue
            vecs = self._get_word_vectors(words)
            if len(vecs) < 2:
                continue
            distances = cosine_distances(vecs)
            np.fill_diagonal(distances, np.nan)
            mean_dist = np.nanmean(distances)
            dat_score = mean_dist * 100
            all_scores.append(dat_score)

        if not all_scores:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"dat_score": 0.0, "valid_samples": 0},
            )

        all_scores = np.array(all_scores)
        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
        n = len(all_scores)

        from cbf.metrics import mean_ci

        ci_low, ci_high = mean_ci(all_scores.tolist())

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "dat_score": mean_score,
                "dat_score_std": std_score,
                "valid_samples": n,
                "total_samples": len(raw.items),
            },
            confidence_intervals={"dat_score": (float(ci_low), float(ci_high))},
            metadata={
                "condition": raw.metadata.get("condition", "unknown"),
                "individual_scores": all_scores.tolist(),
            },
        )


def create_dat_benchmark(config: dict | None = None) -> DATBenchmark:
    if config:
        dat_config = DATConfig(**config)
    else:
        dat_config = DATConfig()
    return DATBenchmark(dat_config)
