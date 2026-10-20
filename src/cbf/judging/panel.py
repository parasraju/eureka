import asyncio
import random
import string
from dataclasses import asdict, dataclass, field

import krippendorff
import numpy as np
from scipy import stats

from cbf.benchmarks import BenchmarkItem, RawResults, Scores
from cbf.providers import LLMProvider, LLMRequest

_PUNCTUATION = string.punctuation + "“”‘’«»–—•"
_MARKDOWN_PREFIX = "*_`~#>•–— \t"
_VALID_WINNERS = {"a": "a", "b": "b", "tie": "tie"}


def _first_token(line: str) -> str:
    cleaned = line.strip().lstrip(_MARKDOWN_PREFIX).strip()
    if not cleaned:
        return ""
    token = cleaned.split()[0]
    token = token.strip(_PUNCTUATION + " \t")
    return token.strip("*_`~")


def parse_winner(text: str) -> str:
    """Extract a strict decision from a judge response.

    Only the first non-empty line is considered. After stripping markdown,
    quotes, and surrounding punctuation, its first token must be exactly
    ``A``, ``B``, or ``TIE`` (case-insensitive). Anything else is ``"invalid"``.
    """
    if not text or not text.strip():
        return "invalid"

    first_line = next((line for line in text.splitlines() if line.strip()), "")
    token = _first_token(first_line)
    return _VALID_WINNERS.get(token.lower(), "invalid")


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
    max_pairs: int = 200


@dataclass
class PairwiseResult:
    item_a_id: str
    item_b_id: str
    winner: str  # "a", "b", "tie", or "invalid"
    judge: str
    position_swapped: bool
    reasoning: str = ""
    raw_response: str = ""


def build_judge_config(raw) -> JudgeConfig:
    """Normalize a config value into a JudgeConfig.

    Accepts a list of ``provider:model`` specs, a dict with
    ``models``/``judge_models`` + optional rubric/temperature/swap_position, or None.
    """
    if raw is None:
        return JudgeConfig(judge_models=[])
    if isinstance(raw, list):
        return JudgeConfig(judge_models=[str(s) for s in raw])
    if isinstance(raw, dict):
        models = raw.get("models", raw.get("judge_models", []))
        return JudgeConfig(
            judge_models=[str(s) for s in models],
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 1000)),
            rubric=raw.get("rubric", "") or "",
            swap_position=bool(raw.get("swap_position", True)),
            max_pairs=int(raw.get("max_pairs", 200)),
        )
    return JudgeConfig(judge_models=[])


def judge_model_name(model_spec: str) -> str:
    """The bare model name from a ``provider:model`` spec (or the spec unchanged)."""
    return model_spec.split(":", 1)[1] if ":" in model_spec else model_spec


def is_self_judge(judge_spec: str, model_under_test: str) -> bool:
    """True when a judge is the same model being tested (self-preference risk)."""
    return judge_model_name(judge_spec) == judge_model_name(model_under_test)


def _item_model(item_id: str) -> str | None:
    return item_id.split("::", 1)[0] if "::" in item_id else None


def _is_cross_model_pair(result: PairwiseResult) -> bool:
    model_a = _item_model(result.item_a_id)
    model_b = _item_model(result.item_b_id)
    return model_a is not None and model_b is not None and model_a != model_b


def _model_of_item_from_ids(results: list[PairwiseResult]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for result in results:
        for item_id in (result.item_a_id, result.item_b_id):
            model = _item_model(item_id)
            if model is not None:
                mapping[item_id] = model
    return mapping


def _raw_item_id(item: dict) -> str:
    return f"p{item.get('prompt_idx', 0)}_g{item.get('generation_idx', 0)}"


class JudgePanel:
    def __init__(self, config: JudgeConfig, provider_registry=None, providers: dict | None = None):
        self.config = config
        self.provider_registry = provider_registry
        self.judges: dict[str, LLMProvider] = dict(providers) if providers else {}

    async def initialize(self):
        if self.judges:
            return
        if self.provider_registry is None:
            raise ValueError("JudgePanel requires a provider_registry or explicit providers")
        for model_spec in self.config.judge_models:
            provider_name = model_spec.split(":", 1)[0]
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
        if swap and winner in {"a", "b"}:
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

    async def _judge_ordered_pairs(
        self,
        texts: dict[str, str],
        pairs: list[tuple[str, str]],
    ) -> list[PairwiseResult]:
        results: list[PairwiseResult] = []
        for judge_model in self.config.judge_models:
            for item_a_id, item_b_id in pairs:
                results.append(await self.judge_pair(
                    judge_model,
                    texts[item_a_id],
                    texts[item_b_id],
                    item_a_id,
                    item_b_id,
                    swap=False,
                ))
                if self.config.swap_position:
                    results.append(await self.judge_pair(
                        judge_model,
                        texts[item_a_id],
                        texts[item_b_id],
                        item_a_id,
                        item_b_id,
                        swap=True,
                    ))
        return results

    async def judge_all_pairs(
        self,
        texts: dict[str, str],
        item_ids: list[str] | None = None,
        max_pairs: int | None = None,
    ) -> list[PairwiseResult]:
        if item_ids is None:
            item_ids = list(texts.keys())

        pairs = [
            (item_ids[i], item_ids[j])
            for i in range(len(item_ids))
            for j in range(i + 1, len(item_ids))
        ]

        cap = self.config.max_pairs if max_pairs is None else int(max_pairs)
        if cap < 0:
            raise ValueError("max_pairs must be non-negative")
        if len(pairs) > cap:
            raise ValueError(
                f"Refusing to judge {len(pairs)} pairs: max_pairs is {cap}. "
                "Increase JudgeConfig.max_pairs or reduce the item set."
            )

        return await self._judge_ordered_pairs(texts, pairs)

    async def judge_models_pairwise(
        self,
        texts_by_model: dict[str, list[str]],
        max_pairs: int = 200,
    ) -> list[PairwiseResult]:
        """Judge deterministically sampled pairs that span different models.

        Item ids are ``"<model>::<idx>"``. At most ``max_pairs`` cross-model
        item pairs are judged (also capped by ``JudgeConfig.max_pairs``).
        """
        texts: dict[str, str] = {}
        for model in sorted(texts_by_model):
            for idx, text in enumerate(texts_by_model[model]):
                texts[f"{model}::{idx}"] = text

        models = sorted(texts_by_model)
        pairs: list[tuple[str, str]] = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a, model_b = models[i], models[j]
                for idx_a in range(len(texts_by_model[model_a])):
                    for idx_b in range(len(texts_by_model[model_b])):
                        pairs.append((f"{model_a}::{idx_a}", f"{model_b}::{idx_b}"))

        cap = min(int(max_pairs), int(self.config.max_pairs))
        if cap < 0:
            raise ValueError("max_pairs must be non-negative")
        if len(pairs) > cap:
            rng = random.Random(0)
            chosen = sorted(rng.sample(range(len(pairs)), cap))
            pairs = [pairs[i] for i in chosen]

        return await self._judge_ordered_pairs(texts, pairs)


def resolve_pairs(results: list[PairwiseResult]) -> list[PairwiseResult]:
    """Collapse repeated judgments of the same pair (e.g. position swaps) into one.

    Invalid judgments are ignored. Agreement across orders -> that winner; any
    disagreement -> a tie; a single decided judgment among ties -> that winner.
    A pair with only invalid judgments resolves to ``"invalid"``.
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
        valid = [r for r in group if r.winner != "invalid"]
        decided = set()
        for r in valid:
            if r.winner == "a":
                decided.add(a_id)
            elif r.winner == "b":
                decided.add(b_id)

        if not valid:
            winner = "invalid"
        elif len(decided) == 1:
            winner_id = next(iter(decided))
            winner = "a" if winner_id == a_id else "b"
        else:
            winner = "tie"

        base = valid[0] if valid else group[0]
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
        if result.winner == "invalid":
            continue
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
        if r.winner == "invalid":
            continue
        i = item_to_idx[r.item_a_id]
        j = item_to_idx[r.item_b_id]
        if r.winner == "a":
            wins[i, j] += 1
        elif r.winner == "b":
            wins[j, i] += 1
        else:
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


def compute_model_matches(results: list[PairwiseResult]) -> dict[str, dict[str, int]]:
    """Per-model wins/losses/ties from cross-model pairs (item ids encode models)."""
    matches: dict[str, dict[str, int]] = {}
    for result in results:
        model_a = _item_model(result.item_a_id)
        model_b = _item_model(result.item_b_id)
        if model_a is None or model_b is None or model_a == model_b:
            continue
        for model in (model_a, model_b):
            matches.setdefault(model, {"wins": 0, "losses": 0, "ties": 0})
        if result.winner == "a":
            matches[model_a]["wins"] += 1
            matches[model_b]["losses"] += 1
        elif result.winner == "b":
            matches[model_b]["wins"] += 1
            matches[model_a]["losses"] += 1
        elif result.winner == "tie":
            matches[model_a]["ties"] += 1
            matches[model_b]["ties"] += 1
    return matches


def compute_model_ratings(
    results: list[PairwiseResult],
    model_of_item: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Elo/BT/win-rate ratings for models, using cross-model pairs only.

    Ties count as half a win in ``win_rate``. Invalid judgments are excluded
    from ratings and reported via ``invalid_rate``.
    """
    mapping = dict(model_of_item)

    def model_for(item_id: str) -> str | None:
        return mapping.get(item_id) or _item_model(item_id)

    cross: list[PairwiseResult] = []
    for result in results:
        model_a = model_for(result.item_a_id)
        model_b = model_for(result.item_b_id)
        if model_a is None or model_b is None or model_a == model_b:
            continue
        mapping.setdefault(result.item_a_id, model_a)
        mapping.setdefault(result.item_b_id, model_b)
        cross.append(result)

    matches: dict[str, dict[str, int]] = {}
    for result in cross:
        model_a = mapping[result.item_a_id]
        model_b = mapping[result.item_b_id]
        for model in (model_a, model_b):
            matches.setdefault(model, {"wins": 0, "losses": 0, "ties": 0})
        if result.winner == "a":
            matches[model_a]["wins"] += 1
            matches[model_b]["losses"] += 1
        elif result.winner == "b":
            matches[model_b]["wins"] += 1
            matches[model_a]["losses"] += 1
        elif result.winner == "tie":
            matches[model_a]["ties"] += 1
            matches[model_b]["ties"] += 1

    win_rate: dict[str, float] = {}
    for model, record in matches.items():
        games = record["wins"] + record["losses"] + record["ties"]
        win_rate[model] = (
            (record["wins"] + 0.5 * record["ties"]) / games if games else 0.0
        )

    model_results: list[PairwiseResult] = []
    for index, result in enumerate(cross):
        if result.winner == "invalid":
            continue
        model_results.append(PairwiseResult(
            item_a_id=mapping[result.item_a_id],
            item_b_id=mapping[result.item_b_id],
            winner=result.winner,
            judge=f"{result.judge}::{index}",
            position_swapped=result.position_swapped,
            reasoning=result.reasoning,
            raw_response=result.raw_response,
        ))

    elo = compute_elo(model_results) if model_results else {}
    bt = compute_bradley_terry(model_results) if model_results else {}
    invalid = sum(1 for r in cross if r.winner == "invalid")
    invalid_rate = invalid / len(cross) if cross else 0.0

    return {
        "elo": elo,
        "bt": bt,
        "win_rate": win_rate,
        "invalid_rate": invalid_rate,
    }


def compute_inter_judge_agreement(results: list[PairwiseResult]) -> dict:
    judges = sorted({r.judge for r in results})
    item_list = sorted({r.item_a_id for r in results} | {r.item_b_id for r in results})
    invalid_rate = (
        sum(1 for r in results if r.winner == "invalid") / len(results)
        if results else 0.0
    )
    if len(judges) < 2:
        return {
            "krippendorff_alpha": 0.0,
            "mean_spearman": 0.0,
            "num_judges": len(judges),
            "num_items": len(item_list),
            "invalid_rate": invalid_rate,
        }

    judge_ratings: dict[str, dict[str, float]] = {}
    rated_items: set[str] = set()
    for judge in judges:
        ratings = compute_elo([
            r for r in results if r.judge == judge and r.winner != "invalid"
        ])
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
                if len(set(x)) < 2 or len(set(y)) < 2:
                    continue
                corr, _ = stats.spearmanr(x, y)
                if not np.isnan(corr):
                    spearman_corrs.append(corr)

    return {
        "krippendorff_alpha": float(alpha) if not np.isnan(alpha) else 0.0,
        "mean_spearman": float(np.mean(spearman_corrs)) if spearman_corrs else 0.0,
        "num_judges": len(judges),
        "num_items": len(item_list),
        "invalid_rate": invalid_rate,
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

    async def judge(self, raw: RawResults, cfg: dict, resolve_provider) -> None:
        """Run the judge panel over this run's texts and store pairwise results.

        ``resolve_provider(model_spec)`` returns a (logging-wrapped) LLMProvider.
        Called by the runner after ``run()``; a no-op when no judges are configured.
        """
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

        judge_config.judge_models = selected
        if not judge_config.rubric:
            judge_config.rubric = self.rubric

        if resolve_provider is None:
            raise ValueError("Judging requires a provider resolver to log judge calls")

        providers = {spec: resolve_provider(spec) for spec in selected}
        panel = JudgePanel(judge_config, providers=providers)
        await panel.initialize()

        texts: dict[str, str] = {}
        for item in raw.items:
            if "error" in item:
                continue
            texts[_raw_item_id(item)] = item.get("text", "")

        pairwise: list[PairwiseResult] = []
        pairs_judged = len(texts) * (len(texts) - 1) // 2
        if len(texts) >= 2:
            pairwise = await panel.judge_all_pairs(texts, list(texts.keys()))

        raw.metadata["pairwise"] = [asdict(p) for p in pairwise]
        raw.metadata["pairs_judged"] = pairs_judged
        raw.metadata["judge_calls"] = len(pairwise)
        raw.metadata["judge"] = {
            "models": selected,
            "filtered_self_judges": filtered,
            "swap_position": judge_config.swap_position,
            "max_pairs": judge_config.max_pairs,
            "pairs_judged": pairs_judged,
            "judge_calls": len(pairwise),
        }

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
                metadata={"calibration": {
                    "available": False,
                    "reason": "no generation items to judge",
                }},
            )

        metrics: dict[str, float] = {
            "items_generated": len(raw.items),
            "prompts_tested": len(by_prompt),
        }
        metadata: dict = {"by_prompt": {str(k): len(v) for k, v in by_prompt.items()}}
        if raw.metadata.get("judge"):
            metadata["judge"] = raw.metadata["judge"]

        pairwise = [PairwiseResult(**p) for p in raw.metadata.get("pairwise", [])]
        cross_raw = raw.metadata.get("cross_model_pairwise", [])
        cross_pairwise = [
            p if isinstance(p, PairwiseResult) else PairwiseResult(**p)
            for p in cross_raw
        ]

        model_of_item = dict(raw.metadata.get("model_of_item") or {})
        if cross_pairwise:
            if not model_of_item:
                model_of_item = _model_of_item_from_ids(cross_pairwise)
            intra_pairwise = pairwise
        elif any(_is_cross_model_pair(r) for r in pairwise):
            cross_pairwise = pairwise
            if not model_of_item:
                model_of_item = _model_of_item_from_ids(pairwise)
            intra_pairwise = [r for r in pairwise if not _is_cross_model_pair(r)]
        else:
            intra_pairwise = pairwise

        if pairwise or cross_pairwise:
            metrics.setdefault("elo_rating", 0.0)
        if intra_pairwise:
            elo = compute_elo(intra_pairwise)
            bt = compute_bradley_terry(intra_pairwise)
            agreement = compute_inter_judge_agreement(intra_pairwise)
            metrics["bt_rating"] = float(np.mean(list(bt.values()))) if bt else 0.0
            metrics["agreement"] = agreement["krippendorff_alpha"]
            metrics["reliable"] = 1.0 if agreement["num_judges"] >= 2 else 0.0
            metrics["invalid_rate"] = agreement["invalid_rate"]
            elo_values = list(elo.values())
            if len(elo_values) >= 2:
                std = float(np.std(elo_values))
                metrics["intra_model_consistency"] = (
                    float(np.mean(elo_values)) / std if std > 0 else 0.0
                )
            metadata["rating_scope"] = "intra_model"

        if cross_pairwise and model_of_item:
            model_ratings = compute_model_ratings(cross_pairwise, model_of_item)
            metadata["model_ratings"] = model_ratings
            metadata["rating_scope"] = "cross_model"
            metadata["model_of_item"] = model_of_item
            if raw.model in model_ratings["elo"]:
                metrics["elo_rating"] = float(model_ratings["elo"][raw.model])
            metrics["invalid_rate"] = float(model_ratings["invalid_rate"])

        if pairwise or cross_pairwise:
            all_pairwise = pairwise + [
                r for r in cross_pairwise if r not in pairwise
            ]
            from cbf.judging.bias import (
                length_controlled_ratings,
                position_win_rate,
                swap_consistency,
            )

            bias: dict = {
                "position_win_rate": position_win_rate(all_pairwise),
                "swap_consistency": swap_consistency(all_pairwise),
            }
            metrics["position_bias"] = abs(bias["position_win_rate"]["overall"] - 0.5)

            if intra_pairwise:
                elo = compute_elo(intra_pairwise)
                lengths: dict[str, int] = {}
                for item in raw.items:
                    if "error" in item:
                        continue
                    lengths[_raw_item_id(item)] = len(item.get("text", ""))
                if sum(1 for item_id in elo if item_id in lengths and lengths[item_id] > 0) >= 3:
                    controlled = length_controlled_ratings(elo, lengths)
                    bias["length_controlled_ratings"] = {
                        item_id: float(value) for item_id, value in controlled.items()
                    }
            metadata["bias"] = bias

        from cbf.judging.calibration import (
            MIN_ITEMS,
            MIN_JUDGES,
            calibrated_panel_scores,
        )

        calibration_results = intra_pairwise or cross_pairwise
        judges = {r.judge for r in calibration_results}
        calibration_items = {
            item_id
            for r in calibration_results
            for item_id in (r.item_a_id, r.item_b_id)
        }
        if len(judges) >= MIN_JUDGES and len(calibration_items) >= MIN_ITEMS:
            try:
                calibration = calibrated_panel_scores(calibration_results)
                calibration["available"] = True
            except Exception as exc:
                calibration = {
                    "available": False,
                    "reason": f"calibration failed: {exc}",
                }
        else:
            calibration = {
                "available": False,
                "reason": (
                    f"requires >= {MIN_JUDGES} judges and >= {MIN_ITEMS} items "
                    f"(have {len(judges)} judges and {len(calibration_items)} items)"
                ),
            }
        metadata["calibration"] = calibration

        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics=metrics,
            metadata=metadata,
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
