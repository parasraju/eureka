"""CREATE-style associative creativity: connect distant concepts with verifiable chains."""

import asyncio
import re
from dataclasses import dataclass, field

import numpy as np

from cbf import metrics as cbf_metrics
from cbf.metrics import bootstrap_ci, distinct_n

from .base import BenchmarkItem, RawResults, Scores


@dataclass
class AssociativeConfig:
    pairs: list[tuple[str, str]] = field(default_factory=lambda: [
        ("ocean", "algorithm"),
        ("volcano", "lullaby"),
        ("telescope", "bread"),
        ("democracy", "crystal"),
        ("monsoon", "chess"),
    ])
    num_hops: int = 4
    num_samples: int = 10
    temperature: float = 0.8
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    connectivity_threshold: float = 0.35


CHAIN_PROMPT = """Connect the concepts "{a}" and "{b}" with a chain of exactly {n} intermediate concepts.
Each consecutive pair must be directly related in the real world (a common association, not a vague link).

Format your answer as exactly {n} lines, one concept per line. No numbering, no explanation."""


def parse_chain(text: str, expected_hops: int, endpoints: tuple[str, str]) -> list[str]:
    lines = [re.sub(r"^[0-9]+[.)]\s*", "", line.strip()).lower() for line in text.strip().splitlines()]
    concepts = [re.sub(r"[^a-z \-]", "", line).strip() for line in lines]
    concepts = [c for c in concepts if c]
    return concepts[:expected_hops]


def validate_chain(concepts: list[str], expected_hops: int, endpoints: tuple[str, str]) -> dict:
    a, b = (e.lower() for e in endpoints)
    valid = len(concepts) == expected_hops
    no_dupes = len(set(concepts)) == len(concepts)
    endpoints_absent = a not in concepts and b not in concepts
    no_empty = all(len(c.split()) <= 4 for c in concepts)
    return {
        "valid": bool(valid and no_dupes and endpoints_absent and no_empty),
        "complete": len(concepts) == expected_hops,
        "no_duplicates": no_dupes,
        "distinct_endpoints": endpoints_absent,
        "n_concepts": len(concepts),
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class AssociativeBenchmark:
    id = "associative"
    version = "1.0.0"

    def __init__(self, config: AssociativeConfig | None = None):
        self.config = config or AssociativeConfig()

    def prompts(self) -> list[BenchmarkItem]:
        items = []
        for i, (a, b) in enumerate(self.config.pairs):
            items.append(BenchmarkItem(
                id=f"assoc_{i}_{a}_{b}",
                prompt=CHAIN_PROMPT.format(a=a, b=b, n=self.config.num_hops),
                metadata={"pair": (a, b), "pair_idx": i},
            ))
        return items

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        from cbf.providers import LLMRequest

        num_samples = cfg.get("num_samples", self.config.num_samples)
        temperature = cfg.get("temperature", self.config.temperature)
        hops = cfg.get("num_hops", self.config.num_hops)

        all_items: list[dict] = []
        call_ids: list[str] = []
        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))

        async def run_sample(pair_idx: int, a: str, b: str, sample_idx: int):
            async with semaphore:
                req = LLMRequest(
                    model=model,
                    messages=[{"role": "user", "content": CHAIN_PROMPT.format(a=a, b=b, n=hops)}],
                    temperature=temperature,
                    max_tokens=120,
                    seed=cfg.get("seed"),
                )
                resp = await provider.complete(req)
                call_ids.append(resp.request_id)
                return {
                    "pair_idx": pair_idx,
                    "pair": [a, b],
                    "sample_idx": sample_idx,
                    "response": resp.text,
                    "call_id": resp.request_id,
                    "latency_s": resp.latency_s,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }

        tasks = []
        for i, (a, b) in enumerate(self.config.pairs):
            for s in range(num_samples):
                tasks.append(run_sample(i, a, b, s))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                all_items.append({"error": str(r), "chain": []})
            else:
                chain = parse_chain(r["response"], hops, tuple(r["pair"]))
                r["chain"] = chain
                r["validation"] = validate_chain(chain, hops, tuple(r["pair"]))
                r["valid"] = r["validation"]["valid"]
                all_items.append(r)

        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=all_items,
            call_ids=call_ids,
            metadata={"config": cfg, "hops": hops},
        )

    def _compute_connectivity(self, valid_items: list[dict]) -> tuple[float, bool]:
        """Fraction of valid chains whose every hop clears the embedding threshold.

        Returns (connectivity, embeddings_unavailable). Unavailable is True only
        when no chain could be embedded at all.
        """
        if not valid_items:
            return 0.0, False

        threshold = self.config.connectivity_threshold
        passed: list[bool] = []
        embedded_any = False

        for item in valid_items:
            pair = item.get("pair") or []
            chain = item.get("chain") or []
            if len(pair) != 2 or not chain:
                passed.append(False)
                continue

            a, b = str(pair[0]).lower(), str(pair[1]).lower()
            sequence = [a] + [str(c).lower() for c in chain] + [b]
            try:
                vectors = np.asarray(
                    cbf_metrics.embed_texts(sequence, self.config.st_model), dtype=float
                )
            except Exception:
                passed.append(False)
                continue

            if vectors.ndim != 2 or len(vectors) != len(sequence):
                passed.append(False)
                continue

            embedded_any = True
            hops_ok = all(
                _cosine_similarity(vectors[i], vectors[i + 1]) >= threshold
                for i in range(len(sequence) - 1)
            )
            passed.append(bool(hops_ok))

        if not embedded_any:
            return 0.0, True
        return float(np.mean(passed)), False

    def score(self, raw: RawResults) -> Scores:
        valid_items = [i for i in raw.items if i.get("valid")]
        ok = [i for i in raw.items if "error" not in i]

        if not ok:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"chain_validity": 0.0, "connectivity": 0.0, "associative_diversity": 0.0},
                metadata={
                    "error": "no successful generations",
                    "connectivity_method": "embedding-threshold",
                    "connectivity_threshold": self.config.connectivity_threshold,
                },
            )

        validity = len(valid_items) / len(ok)
        no_dupes = float(np.mean([1.0 if i["validation"]["no_duplicates"] else 0.0 for i in ok]))

        connectivity, connectivity_unavailable = self._compute_connectivity(valid_items)

        all_concepts = [c for i in valid_items for c in i.get("chain", [])]
        concept_novelty = distinct_n([" ".join(i.get("chain", [])) for i in valid_items], 2) if valid_items else 0.0

        diversity = 0.0
        if len(valid_items) >= 2:
            try:
                embeddings = cbf_metrics.embed_texts(
                    [" ".join(i["chain"]) for i in valid_items], self.config.st_model
                )
                diversity = cbf_metrics.mean_pairwise_distance(embeddings)
            except Exception:
                diversity = 0.0

        per_sample_validity = [1.0 if i.get("valid") else 0.0 for i in ok]
        ci = bootstrap_ci(per_sample_validity)

        metadata = {
            "unique_concepts": len(set(all_concepts)),
            "connectivity_method": "embedding-threshold",
            "connectivity_threshold": self.config.connectivity_threshold,
        }
        if connectivity_unavailable:
            metadata["connectivity_unavailable"] = True

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "chain_validity": validity,
                "connectivity": connectivity,
                "no_repeats": no_dupes,
                "associative_diversity": float(diversity),
                "concept_novelty": float(concept_novelty),
                "valid_samples": len(valid_items),
                "total_samples": len(ok),
            },
            confidence_intervals={"chain_validity": ci},
            metadata=metadata,
        )


def create_associative_benchmark(config: dict | None = None) -> AssociativeBenchmark:
    cfg = AssociativeConfig(**config) if config else AssociativeConfig()
    return AssociativeBenchmark(cfg)
