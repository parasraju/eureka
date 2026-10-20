import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from .base import BenchmarkItem, RawResults, Scores

_GLOVE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _get_glove_vectors(config: "DATConfig") -> dict[str, np.ndarray]:
    """Cache GloVe vectors per config (by glove path or default cache dir)."""
    cache_key = config.glove_path or "default"
    if cache_key in _GLOVE_CACHE:
        return _GLOVE_CACHE[cache_key]

    if config.glove_path:
        path = Path(config.glove_path)
    else:
        cache_dir = Path.home() / ".cache" / "cbf" / "glove"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "glove.6B.300d.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"GloVe vectors not found at {path}. Download from https://nlp.stanford.edu/projects/glove/ "
            f"or set DATConfig.glove_path"
        )

    vectors: dict[str, np.ndarray] = {}
    dim = 300
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == dim + 1:
                word = parts[0]
                vectors[word] = np.array([float(x) for x in parts[1:]], dtype=np.float32)

    _GLOVE_CACHE[cache_key] = vectors
    return vectors


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
                raise r
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

    def _get_word_vectors(self, words: list[str]) -> np.ndarray:
        if self.config.embedding_model == "glove":
            glove = _get_glove_vectors(self.config)
            return np.array([glove[w] for w in words if w in glove], dtype=np.float32)
        raise NotImplementedError(f"Embedding model {self.config.embedding_model} not implemented")

    def score(self, raw: RawResults) -> Scores:
        from cbf.metrics import mean_ci

        all_scores: list[float] = []
        valid_samples = 0
        oov_terms = 0

        for item in raw.items:
            if not item.get("valid", False):
                all_scores.append(0.0)
                continue

            words = item.get("words", [])
            if len(words) < 2:
                all_scores.append(0.0)
                continue

            vecs = self._get_word_vectors(words)
            oov_terms += max(0, len(words) - len(vecs))
            if len(vecs) < 2:
                all_scores.append(0.0)
                continue

            distances = cosine_distances(vecs)
            np.fill_diagonal(distances, np.nan)
            mean_dist = float(np.nanmean(distances))
            if np.isnan(mean_dist):
                all_scores.append(0.0)
                continue

            all_scores.append(mean_dist * 100)
            valid_samples += 1

        total_samples = len(raw.items)
        scores = np.asarray(all_scores, dtype=float)
        mean_score = float(np.mean(scores)) if scores.size else 0.0
        std_score = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
        compliance_rate = (valid_samples / total_samples) if total_samples else 0.0
        ci_low, ci_high = mean_ci(scores.tolist())

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "dat_score": mean_score,
                "dat_score_std": std_score,
                "valid_samples": valid_samples,
                "total_samples": total_samples,
                "compliance_rate": compliance_rate,
                "oov_terms": oov_terms,
            },
            confidence_intervals={"dat_score": (float(ci_low), float(ci_high))},
            metadata={
                "condition": raw.metadata.get("condition", "unknown"),
                "individual_scores": scores.tolist(),
            },
        )


def create_dat_benchmark(config: dict | None = None) -> DATBenchmark:
    if config:
        dat_config = DATConfig(**config)
    else:
        dat_config = DATConfig()
    return DATBenchmark(dat_config)
