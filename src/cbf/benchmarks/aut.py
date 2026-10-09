import re
import asyncio
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from .base import Benchmark, BenchmarkItem, RawResults, Scores
from .dat import DATBenchmark, DATConfig


@dataclass
class AUTConfig:
    num_uses: int = 10
    num_samples: int = 30
    temperature: float = 0.7
    objects: list[str] = field(default_factory=lambda: [
        "brick", "paperclip", "newspaper", "towel", "shoe", "pencil", "cup", "box", "rope", "button"
    ])
    embedding_model: str = "glove"
    glove_path: str | None = None


class AUTBenchmark:
    id = "aut"
    version = "1.0.0"

    AUT_PROMPT_NEUTRAL = """List {num_uses} creative and unusual uses for a {object}.
Format as a numbered list (1., 2., etc.).
Only list the uses, one per line."""

    AUT_PROMPT_CREATIVE = """Be as creative and original as possible. List {num_uses} creative and unusual uses for a {object}.
Format as a numbered list (1., 2., etc.).
Only list the uses, one per line."""

    COMMON_USES = {
        "brick": ["building", "construction", "wall", "house", "pathway"],
        "paperclip": ["holding papers", "clipping", "organizing documents", "fastening"],
        "newspaper": ["reading", "news", "information", "wrapping", "lining"],
        "towel": ["drying", "wiping", "bath", "beach", "cleaning"],
        "shoe": ["wearing", "walking", "running", "footwear", "protection"],
        "pencil": ["writing", "drawing", "sketching", "marking"],
        "cup": ["drinking", "holding liquid", "beverage", "mug"],
        "box": ["storage", "packing", "moving", "container"],
        "rope": ["tying", "climbing", "pulling", "securing", "binding"],
        "button": ["fastening", "clothing", "shirt", "pressing", "switch"],
    }

    def __init__(self, config: AUTConfig | None = None):
        self.config = config or AUTConfig()
        self._dat = DATBenchmark(DATConfig(embedding_model=self.config.embedding_model, glove_path=self.config.glove_path))

    def prompts(self) -> list[BenchmarkItem]:
        items = []
        for obj in self.config.objects:
            items.append(BenchmarkItem(
                id=f"aut_{obj}_neutral",
                prompt=self.AUT_PROMPT_NEUTRAL.format(num_uses=self.config.num_uses, object=obj),
                metadata={"condition": "neutral", "object": obj},
            ))
            items.append(BenchmarkItem(
                id=f"aut_{obj}_creative",
                prompt=self.AUT_PROMPT_CREATIVE.format(num_uses=self.config.num_uses, object=obj),
                metadata={"condition": "creative", "object": obj},
            ))
        return items

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        from cbf.providers import LLMRequest

        num_samples = cfg.get("num_samples", self.config.num_samples)
        temperature = cfg.get("temperature", self.config.temperature)
        condition = cfg.get("condition", "neutral")

        prompt_template = self.AUT_PROMPT_CREATIVE if condition == "creative" else self.AUT_PROMPT_NEUTRAL

        all_items = []
        call_ids = []

        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))

        async def run_sample(object_name: str, sample_idx: int):
            async with semaphore:
                prompt = prompt_template.format(num_uses=self.config.num_uses, object=object_name)
                req = LLMRequest(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=300,
                    seed=cfg.get("seed"),
                )
                resp = await provider.complete(req)
                call_ids.append(resp.request_id)
                return {
                    "object": object_name,
                    "sample_idx": sample_idx,
                    "response": resp.text,
                    "call_id": resp.request_id,
                    "latency_s": resp.latency_s,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }

        tasks = []
        for obj in self.config.objects:
            for i in range(num_samples):
                tasks.append(run_sample(obj, i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                all_items.append({"object": "unknown", "sample_idx": -1, "error": str(r), "uses": []})
            else:
                uses = self._parse_uses(r["response"])
                r["uses"] = uses
                r["valid"] = len(uses) > 0
                all_items.append(r)

        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=all_items,
            call_ids=call_ids,
            metadata={"config": cfg, "condition": condition},
        )

    def _parse_uses(self, text: str) -> list[str]:
        lines = text.strip().split('\n')
        uses = []
        for line in lines:
            line = line.strip()
            line = re.sub(r'^\d+[.)]\s*', '', line)
            line = line.strip()
            if line and len(line) > 3:
                uses.append(line.lower())
        return uses[:self.config.num_uses]

    def _get_use_vectors(self, uses: list[str]) -> np.ndarray:
        return self._dat._get_word_vectors(uses)

    def _compute_flexibility(self, uses: list[str], object_name: str) -> int:
        common = self.COMMON_USES.get(object_name, [])
        common_vecs = self._get_use_vectors(common) if common else np.array([])
        use_vecs = self._get_use_vectors(uses)

        if len(common_vecs) == 0 or len(use_vecs) == 0:
            return len(set(uses))

        distances = cosine_distances(use_vecs, common_vecs)
        min_dists = np.min(distances, axis=1)
        novel = np.sum(min_dists > 0.7)
        return int(novel) + len(set(uses)) // 2

    def score(self, raw: RawResults) -> Scores:
        valid_items = [item for item in raw.items if item.get("valid", False)]
        if not valid_items:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"fluency": 0.0, "flexibility": 0.0, "originality": 0.0, "valid_samples": 0},
            )

        by_object: dict[str, list[dict]] = {}
        for item in valid_items:
            obj = item.get("object", "unknown")
            if obj not in by_object:
                by_object[obj] = []
            by_object[obj].append(item)

        all_fluency = []
        all_flexibility = []
        all_originality = []

        for obj, items in by_object.items():
            for item in items:
                uses = item.get("uses", [])
                if not uses:
                    continue

                fluency = len(uses)
                all_fluency.append(fluency)

                flexibility = self._compute_flexibility(uses, obj)
                all_flexibility.append(flexibility)

                use_vecs = self._get_use_vectors(uses)
                if len(use_vecs) > 1:
                    distances = cosine_distances(use_vecs)
                    np.fill_diagonal(distances, np.nan)
                    originality = float(np.nanmean(distances)) * 100
                else:
                    originality = 0.0
                all_originality.append(originality)

        from scipy import stats

        def ci(data):
            if len(data) < 2:
                return (float(np.mean(data)) if data else 0.0, float(np.mean(data)) if data else 0.0)
            mean = float(np.mean(data))
            sem = stats.sem(data)
            low, high = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
            return (float(low), float(high))

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "fluency": float(np.mean(all_fluency)) if all_fluency else 0.0,
                "flexibility": float(np.mean(all_flexibility)) if all_flexibility else 0.0,
                "originality": float(np.mean(all_originality)) if all_originality else 0.0,
                "elaboration": float(np.mean([len(u) for item in valid_items for u in item.get("uses", [])])) if valid_items else 0.0,
                "valid_samples": len(valid_items),
                "total_samples": len(raw.items),
            },
            confidence_intervals={
                "fluency": ci(all_fluency),
                "flexibility": ci(all_flexibility),
                "originality": ci(all_originality),
            },
            metadata={
                "condition": raw.metadata.get("condition", "unknown"),
                "by_object": {obj: len(items) for obj, items in by_object.items()},
            },
        )


def create_aut_benchmark(config: dict | None = None) -> AUTBenchmark:
    if config:
        aut_config = AUTConfig(**config)
    else:
        aut_config = AUTConfig()
    return AUTBenchmark(aut_config)