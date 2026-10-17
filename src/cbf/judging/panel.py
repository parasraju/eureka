import asyncio
import re
from dataclasses import dataclass, field

import krippendorff
import numpy as np
from scipy import stats

from cbf.benchmarks import BenchmarkItem, RawResults, Scores
from cbf.providers import LLMProvider, LLMRequest

_WINNER_RE = re.compile(r"\b(A|B|TIE)\b")
_WINNER_RE_CI = re.compile(r"\b(A|B|TIE)\b", re.IGNORECASE)
_LEADING_WINNER_RE = re.compile(r"^[^A-Za-z0-9]*(A|B|TIE)\b", re.IGNORECASE)


def parse_winner(text: str) -> str:
    """Extract a decision from a judge response as "a", "b", or "tie".

    Tries, in order: a leading token in the first line, then an uppercase
    standalone token (judges are told to answer with ONLY A/B/TIE), then a
    case-insensitive standalone token. Undecidable responses are ties.
    """
    if not text or not text.strip():
        return "tie"
    stripped = text.strip()
    first_line = stripped.splitlines()[0]

    match = _LEADING_WINNER_RE.search(first_line)
    if not match:
        match = _WINNER_RE.search(first_line) or _WINNER_RE.search(stripped)
    if not match:
        match = _WINNER_RE_CI.search(first_line) or _WINNER_RE_CI.search(stripped)
    if not match:
        return "tie"
    return {"A": "a", "B": "b", "TIE": "tie"}[match.group(1).upper()]


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

        winner = parse_winner(resp.text)
        if swap and winner != "tie":
            winner = "b" if winner == "a" else "a"

        return PairwiseResult(
            item_a_id=item_a_id,
            item_b_id=item_b_id,
            winner=winner,
            judge=judge_model,
            position_swapped=swap,
            reasoning=resp.text.strip(),
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


def resolve_pairs(results: list[PairwiseResult]) -> list[PairwiseResult]:
    """Collapse repeated judgments of the same pair (e.g. position swaps) into one.

    Agreement across orders -> that winner; any disagreement -> a tie; a single
    decided judgment among ties -> that winner. Returns results in first-seen order.
    """
    groups: dict[tuple, list[PairwiseResult]] = {}
    order: list[tuple] = []
    for r in results:
        key = (tuple(sorted((r.item_a_id, r.item_b_id))), r.judge)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    resolved: list[PairwiseResult] = []
    for key in order:
        group = groups[key]
        a_id, b_id = key[0]
        decided = set()
        for r in group:
            if r.winner == "a":
                decided.add(a_id)
            elif r.winner == "b":
                decided.add(b_id)
        if len(decided) == 1:
            winner_id = next(iter(decided))
        else:
            winner_id = None

        winner = "a" if winner_id == a_id else "b" if winner_id == b_id else "tie"
        base = group[0]
        resolved.append(PairwiseResult(
            item_a_id=a_id,
            item_b_id=b_id,
            winner=winner,
            judge=key[1],
            position_swapped=False,
            reasoning=base.reasoning,
            raw_response=base.raw_response,
        ))
    return resolved


def compute_elo(
    results: list[PairwiseResult],
    k_factor: float = 32.0,
    initial_rating: float = 1000.0,
) -> dict[str, float]:
    items = set()
    for r in results:
        items.add(r.item_a_id)
        items.add(r.item_b_id)

    ratings = dict.fromkeys(items, initial_rating)

    for result in resolve_pairs(results):
        a_id, b_id = result.item_a_id, result.item_b_id
        r_a, r_b = ratings[a_id], ratings[b_id]

        e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
        e_b = 1 - e_a

        if result.winner == "a":
            s_a, s_b = 1.0, 0.0
        elif result.winner == "b":
            s_a, s_b = 0.0, 1.0
        else:
            s_a, s_b = 0.5, 0.5

        ratings[a_id] += k_factor * (s_a - e_a)
        ratings[b_id] += k_factor * (s_b - e_b)

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
    for r in resolve_pairs(results):
        i = item_to_idx[r.item_a_id]
        j = item_to_idx[r.item_b_id]
        if r.winner == "a":
            wins[i, j] += 1
        elif r.winner == "b":
            wins[j, i] += 1
        else:
            # Ties count as half a win to each side (Davidson-style smoothing).
            wins[i, j] += 0.5
            wins[j, i] += 0.5

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
    judges = sorted({r.judge for r in results})
    item_list = sorted({r.item_a_id for r in results} | {r.item_b_id for r in results})
    if len(judges) < 2:
        return {
            "krippendorff_alpha": 0.0,
            "mean_spearman": 0.0,
            "num_judges": len(judges),
            "num_items": len(item_list),
        }

    judge_ratings: dict[str, dict[str, float]] = {}
    rated_items: set[str] = set()
    for judge in judges:
        ratings = compute_elo([r for r in results if r.judge == judge])
        judge_ratings[judge] = ratings
        rated_items |= set(ratings)
    item_list = sorted(rated_items)

    matrix = np.array([[judge_ratings[j].get(item, np.nan) for item in item_list] for j in judges])
    try:
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
    except Exception:
        alpha = np.nan

    common = [it for it in item_list if all(it in judge_ratings[j] for j in judges)]
    spearman_corrs = []
    if len(common) >= 2:
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                x = [judge_ratings[judges[i]][it] for it in common]
                y = [judge_ratings[judges[j]][it] for it in common]
                corr, _ = stats.spearmanr(x, y)
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
                model=model.split(":", 1)[1] if ":" in model else model,
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
