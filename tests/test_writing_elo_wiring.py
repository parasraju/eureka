import asyncio
import sys

sys.path.insert(0, r'D:\p\eureka\src')

from cbf.judging.panel import (
    WritingEloBenchmark,
    build_judge_config,
    is_self_judge,
)
from cbf.providers.base import LLMResponse


class FakeGenProvider:
    name = "fake"

    async def complete(self, req):
        prompt = req.messages[0]["content"]
        return LLMResponse(
            text=f"story about {prompt[:20]}",
            model=req.model,
            request_id="gen",
            input_tokens=1,
            output_tokens=1,
            latency_s=0.0,
            raw={},
        )


class FakeJudgeProvider:
    name = "fake"

    def __init__(self, verdict):
        self.verdict = verdict
        self.seen = []

    async def complete(self, req):
        self.seen.append(req.messages[0]["content"])
        return LLMResponse(
            text=self.verdict,
            model=req.model,
            request_id="judge",
            input_tokens=1,
            output_tokens=1,
            latency_s=0.0,
            raw={},
        )


def test_build_judge_config_forms():
    assert build_judge_config(["openai:gpt-4o"]).judge_models == ["openai:gpt-4o"]
    cfg = build_judge_config({"models": ["a:b"], "rubric": "r", "swap_position": False})
    assert cfg.judge_models == ["a:b"]
    assert cfg.rubric == "r"
    assert cfg.swap_position is False
    assert build_judge_config(None).judge_models == []


def test_is_self_judge():
    assert is_self_judge("openai:gpt-4o", "gpt-4o")
    assert is_self_judge("gpt-4o", "gpt-4o")
    assert not is_self_judge("anthropic:claude-3-5-sonnet", "gpt-4o")


def test_run_then_judge_then_score():
    bench = WritingEloBenchmark({"prompts": ["p1", "p2"]})
    raw = asyncio.run(bench.run(FakeGenProvider(), "model-x", {"k_generations": 1}))
    assert len(raw.items) == 2

    judge = FakeJudgeProvider("A")
    cfg = {"judges": {"models": ["fake:judge-1", "fake:judge-2"]}}
    asyncio.run(bench.judge(raw, cfg, lambda spec: judge))

    assert raw.metadata["pairwise"]
    assert len(raw.metadata["pairwise"]) == 4  # 2 judges x 2 orders
    assert len(judge.seen) == 4

    scores = bench.score(raw)
    assert "bt_rating" in scores.metrics
    assert "elo_rating" in scores.metrics
    assert "agreement" in scores.metrics
    assert scores.metrics["reliable"] == 1.0


def test_self_judge_only_raises():
    bench = WritingEloBenchmark({"prompts": ["p1", "p2"]})
    raw = asyncio.run(bench.run(FakeGenProvider(), "gpt-4o", {"k_generations": 1}))
    cfg = {"judges": ["openai:gpt-4o"]}
    raised = False
    try:
        asyncio.run(bench.judge(raw, cfg, lambda spec: FakeJudgeProvider("A")))
    except ValueError:
        raised = True
    assert raised


def test_self_judge_filtered_others_kept():
    bench = WritingEloBenchmark({"prompts": ["p1", "p2"]})
    raw = asyncio.run(bench.run(FakeGenProvider(), "gpt-4o", {"k_generations": 1}))
    cfg = {"judges": ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022"]}
    asyncio.run(bench.judge(raw, cfg, lambda spec: FakeJudgeProvider("A")))
    assert raw.metadata["judge"]["filtered_self_judges"] == ["openai:gpt-4o"]
    assert raw.metadata["judge"]["models"] == ["anthropic:claude-3-5-sonnet-20241022"]


def test_judge_noop_without_judges():
    bench = WritingEloBenchmark({"prompts": ["p1"]})
    raw = asyncio.run(bench.run(FakeGenProvider(), "m", {"k_generations": 1}))
    asyncio.run(bench.judge(raw, {}, lambda spec: FakeJudgeProvider("A")))
    assert "pairwise" not in raw.metadata
    scores = bench.score(raw)
    assert scores.metrics["items_generated"] == 1
    assert scores.metrics["prompts_tested"] == 1


if __name__ == "__main__":
    test_build_judge_config_forms()
    test_is_self_judge()
    test_run_then_judge_then_score()
    test_self_judge_only_raises()
    test_self_judge_filtered_others_kept()
    test_judge_noop_without_judges()
    print("All writing_elo wiring tests passed!")
