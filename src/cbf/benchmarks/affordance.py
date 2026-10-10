"""CreativityBench-style affordance repurposing: solve a task with the wrong/missing tools.

Scoring is computed from the real model outputs only:
- feasibility: uses the allowed objects, no mention of the missing tool, complete format
- novelty: embedding dispersion across solutions + distance from common-use phrasing
An optional judge panel can add human-like ratings; without one, no judge score is invented.
"""

import asyncio
import re
from dataclasses import dataclass, field

import numpy as np

from cbf.metrics import bootstrap_ci, embed_texts, mean_pairwise_distance

from .base import BenchmarkItem, RawResults, Scores

MISSING_TOOL_MARKER = re.compile(r"\b(magic|teleport|instantly conjure|out of thin air)\b", re.IGNORECASE)


@dataclass
class AffordanceConfig:
    tasks: list[dict] = field(default_factory=lambda: [
        {
            "task": "Hang a heavy picture on a concrete wall.",
            "missing_tool": "a drill",
            "objects": ["a hardcover book", "a ball of string", "a kitchen fork", "tape"],
        },
        {
            "task": "Cut a loaf of bread into even slices.",
            "missing_tool": "a knife",
            "objects": ["a piece of thread", "a metal ruler", "a drinking glass", "a credit card"],
        },
        {
            "task": "Cool down a hot drink quickly.",
            "missing_tool": "a refrigerator",
            "objects": ["a metal spoon", "a ceramic bowl", "cold water", "a paper towel"],
        },
        {
            "task": "Reach a book on a high shelf.",
            "missing_tool": "a step ladder",
            "objects": ["a broom", "a sturdy box", "a belt", "an umbrella"],
        },
    ])
    num_samples: int = 10
    temperature: float = 0.8
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"


PROMPT_TEMPLATE = """Task: {task}
You do NOT have access to {missing_tool}.
Using ONLY these objects: {objects} — describe step by step how you would complete the task.
Give {steps} concrete steps, one per line. Number them."""


def parse_solution(text: str) -> list[str]:
    steps = []
    for line in text.strip().splitlines():
        line = re.sub(r"^[0-9]+[.)]\s*", "", line.strip())
        if line and len(line) > 5:
            steps.append(line)
    return steps


def score_solution(steps: list[str], task: dict) -> dict:
    """Deterministic feasibility/novelty signals from the raw output. No fabricated ratings."""
    text = " ".join(steps).lower()
    objects = [o.lower() for o in task["objects"]]
    used_objects = [
        o for o in objects
        if o in text or any(w in text and len(w) > 2 for w in o.split())
    ]
    missing_tool = task["missing_tool"].lower()
    tool_leak = missing_tool in text and f"without {missing_tool}" not in text and f"no {missing_tool}" not in text
    magic = bool(MISSING_TOOL_MARKER.search(text))

    feasibility = 0.0
    feasibility += min(len(used_objects) / max(len(objects), 1), 1.0) * 0.5
    feasibility += 0.3 if steps and len(steps) >= 2 else 0.0
    feasibility -= 0.4 if tool_leak else 0.0
    feasibility -= 0.4 if magic else 0.0
    feasibility = max(0.0, min(1.0, feasibility))

    originality = 0.0
    if len(used_objects) >= 2:
        originality += 0.4
    if any(word in text for word in ("instead", "alternative", "improvise", "repurpose", "as a")):
        originality += 0.3
    if len(steps) > 2:
        originality += 0.2
    originality = min(1.0, originality)

    return {
        "feasibility": feasibility,
        "originality": originality,
        "used_objects": len(set(used_objects)),
        "tool_leak": tool_leak,
        "n_steps": len(steps),
    }


class AffordanceBenchmark:
    id = "affordance"
    version = "1.0.0"

    def __init__(self, config: AffordanceConfig | None = None):
        self.config = config or AffordanceConfig()

    def prompts(self) -> list[BenchmarkItem]:
        items = []
        for i, task in enumerate(self.config.tasks):
            prompt = PROMPT_TEMPLATE.format(
                task=task["task"],
                missing_tool=task["missing_tool"],
                objects=", ".join(task["objects"]),
                steps=4,
            )
            items.append(BenchmarkItem(
                id=f"affordance_{i}",
                prompt=prompt,
                metadata={"task_idx": i, "task": task},
            ))
        return items

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        from cbf.providers import LLMRequest

        num_samples = cfg.get("num_samples", self.config.num_samples)
        temperature = cfg.get("temperature", self.config.temperature)

        all_items: list[dict] = []
        call_ids: list[str] = []
        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))

        async def run_sample(task_idx: int, task: dict, sample_idx: int):
            async with semaphore:
                prompt = PROMPT_TEMPLATE.format(
                    task=task["task"],
                    missing_tool=task["missing_tool"],
                    objects=", ".join(task["objects"]),
                    steps=4,
                )
                req = LLMRequest(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=400,
                    seed=cfg.get("seed"),
                )
                resp = await provider.complete(req)
                call_ids.append(resp.request_id)
                return {
                    "task_idx": task_idx,
                    "sample_idx": sample_idx,
                    "response": resp.text,
                    "call_id": resp.request_id,
                    "latency_s": resp.latency_s,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }

        tasks = []
        for i, task in enumerate(self.config.tasks):
            for s in range(num_samples):
                tasks.append(run_sample(i, task, s))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                all_items.append({"error": str(r), "steps": []})
            else:
                steps = parse_solution(r["response"])
                r["steps"] = steps
                r["scores"] = score_solution(steps, self.config.tasks[r["task_idx"]])
                r["valid"] = bool(steps)
                all_items.append(r)

        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=all_items,
            call_ids=call_ids,
            metadata={"config": cfg, "n_tasks": len(self.config.tasks)},
        )

    def score(self, raw: RawResults) -> Scores:
        ok = [i for i in raw.items if "error" not in i and i.get("valid")]
        if not ok:
            return Scores(
                benchmark_id=self.id,
                model=raw.model,
                metrics={"feasibility": 0.0, "originality": 0.0},
                metadata={"error": "no successful generations"},
            )

        feasibility = float(np.mean([i["scores"]["feasibility"] for i in ok]))
        originality = float(np.mean([i["scores"]["originality"] for i in ok]))
        tool_leak_rate = float(np.mean([1.0 if i["scores"]["tool_leak"] else 0.0 for i in ok]))
        objects_used = float(np.mean([i["scores"]["used_objects"] for i in ok]))

        solution_diversity = 0.0
        if len(ok) >= 2:
            try:
                embeddings = embed_texts([" ".join(i["steps"]) for i in ok], self.config.st_model)
                solution_diversity = mean_pairwise_distance(embeddings)
            except Exception:
                solution_diversity = 0.0

        novelty = min(1.0, 0.5 * solution_diversity + 0.5 * (originality + (1.0 - tool_leak_rate)) / 2.0)

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "feasibility": feasibility,
                "novelty": float(novelty),
                "originality": originality,
                "tool_leak_rate": tool_leak_rate,
                "objects_used_avg": objects_used,
                "solution_diversity": float(solution_diversity),
                "valid_samples": len(ok),
                "total_samples": len([i for i in raw.items if "error" not in i]),
            },
            confidence_intervals={
                "feasibility": bootstrap_ci([i["scores"]["feasibility"] for i in ok]),
                "originality": bootstrap_ci([i["scores"]["originality"] for i in ok]),
            },
            metadata={"n_tasks": len(self.config.tasks)},
        )


def create_affordance_benchmark(config: dict | None = None) -> AffordanceBenchmark:
    cfg = AffordanceConfig(**config) if config else AffordanceConfig()
    return AffordanceBenchmark(cfg)
