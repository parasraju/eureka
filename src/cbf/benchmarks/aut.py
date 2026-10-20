import asyncio
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from cbf import metrics as cbf_metrics

from .base import BenchmarkItem, RawResults, Scores


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
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    flexibility_threshold: float = 0.75


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
        """Embed uses as full phrases; word-level GloVe cannot represent phrases."""
        try:
            vectors = cbf_metrics.embed_texts(uses, self.config.st_model)
        except Exception as exc:
            raise RuntimeError(
                f"AUT embeddings failed (model={self.config.st_model!r}): {exc}"
            ) from exc
        return np.asarray(vectors, dtype=float)

    def _compute_flexibility(self, uses: list[str], object_name: str | None = None) -> int:
        if not uses:
            return 0
        use_vecs = self._get_use_vectors(uses)
        if len(use_vecs) == 0:
            return 0
        return len(cbf_metrics.cluster_by_similarity(use_vecs, self.config.flexibility_threshold))

    def _compute_originality(self, uses: list[str], object_name: str) -> float:
        common = self.COMMON_USES.get(object_name, [])
        if not uses or not common:
            return 0.0

        common_vecs = self._get_use_vectors(common)
        use_vecs = self._get_use_vectors(uses)
        if len(common_vecs) == 0 or len(use_vecs) == 0:
            return 0.0

        centroid = np.mean(common_vecs, axis=0, keepdims=True)
        distances = cosine_distances(use_vecs, centroid).ravel()
        lo = float(np.min(distances))
        hi = float(np.max(distances))
        if hi - lo <= 1e-12:
            return 0.0
        scaled = (distances - lo) / (hi - lo) * 100.0
        return float(np.mean(np.clip(scaled, 0.0, None)))

    def score(self, raw: RawResults) -> Scores:
        scored_items = [item for item in raw.items if "error" not in item]
        valid_items = [item for item in scored_items if item.get("valid", False)]
        if not scored_items:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={
                    "fluency": 0.0,
                    "flexibility": 0.0,
                    "originality": 0.0,
                    "elaboration": 0.0,
                    "valid_samples": 0,
                    "total_samples": len(raw.items),
                    "compliance_rate": 0.0,
                    "error_samples": len(raw.items),
                },
            )

        by_object: dict[str, list[dict]] = {}
        for item in scored_items:
            obj = item.get("object", "unknown")
            if obj not in by_object:
                by_object[obj] = []
            by_object[obj].append(item)

        all_fluency = []
        all_flexibility = []
        all_originality = []
        elaboration_per_sample = []

        for obj, items in by_object.items():
            for item in items:
                uses = item.get("uses", [])
                if not item.get("valid", False) or not uses:
                    # Non-compliant samples score zero rather than disappearing.
                    all_fluency.append(0.0)
                    all_flexibility.append(0.0)
                    all_originality.append(0.0)
                    elaboration_per_sample.append(0.0)
                    continue

                all_fluency.append(len(uses))
                all_flexibility.append(self._compute_flexibility(uses, obj))
                all_originality.append(self._compute_originality(uses, obj))
                elaboration_per_sample.append(float(np.mean([len(u.split()) for u in uses])))

        elaboration = float(np.mean(elaboration_per_sample)) if elaboration_per_sample else 0.0
        compliance_rate = len(valid_items) / len(scored_items) if scored_items else 0.0

        ci = cbf_metrics.mean_ci

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "fluency": float(np.mean(all_fluency)) if all_fluency else 0.0,
                "flexibility": float(np.mean(all_flexibility)) if all_flexibility else 0.0,
                "originality": float(np.mean(all_originality)) if all_originality else 0.0,
                "elaboration": elaboration,
                "valid_samples": len(valid_items),
                "total_samples": len(raw.items),
                "compliance_rate": compliance_rate,
                "error_samples": len(raw.items) - len(scored_items),
            },
            confidence_intervals={
                "fluency": ci(all_fluency),
                "flexibility": ci(all_flexibility),
                "originality": ci(all_originality),
                "elaboration": ci(elaboration_per_sample),
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
