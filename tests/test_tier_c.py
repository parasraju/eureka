import sys

sys.path.insert(0, r'D:\p\eureka\src')

import pytest

from cbf.benchmarks.adapters import (
    AdapterApprovalError,
    RegistryAdapter,
    build_prompts_from_entry,
    load_adapter,
    require_approved,
)
from cbf.benchmarks.affordance import (
    AffordanceBenchmark,
    AffordanceConfig,
    parse_solution,
    score_solution,
)
from cbf.benchmarks.associative import (
    AssociativeBenchmark,
    AssociativeConfig,
    parse_chain,
    validate_chain,
)
from cbf.discovery.registry_db import RegistryEntry


def make_assoc_result_items():
    return [
        {
            "pair_idx": 0,
            "pair": ["ocean", "algorithm"],
            "sample_idx": 0,
            "chain": ["salt", "code", "data", "sort"],
            "validation": {"valid": True, "complete": True, "no_duplicates": True,
                           "distinct_endpoints": True, "n_concepts": 4},
            "valid": True,
        },
        {
            "pair_idx": 0,
            "pair": ["ocean", "algorithm"],
            "sample_idx": 1,
            "chain": ["water"],
            "validation": {"valid": False, "complete": False, "no_duplicates": True,
                           "distinct_endpoints": True, "n_concepts": 1},
            "valid": False,
        },
    ]


def test_parse_chain_strips_numbering():
    chain = parse_chain("1. salt\n2. code\n3. data\n4. sort", 4, ("ocean", "algorithm"))
    assert chain == ["salt", "code", "data", "sort"]


def test_validate_chain():
    assert validate_chain(["salt", "code", "data", "sort"], 4, ("ocean", "algorithm"))["valid"] is True
    assert validate_chain(["ocean", "code", "data", "sort"], 4, ("ocean", "algorithm"))["valid"] is False  # endpoint leak
    assert validate_chain(["salt", "salt", "data", "sort"], 4, ("ocean", "algorithm"))["valid"] is False  # dupes
    assert validate_chain(["salt", "code", "data"], 4, ("ocean", "algorithm"))["valid"] is False  # incomplete


def test_associative_prompts():
    bench = AssociativeBenchmark(AssociativeConfig(pairs=[("ocean", "algorithm")]))
    prompts = bench.prompts()
    assert len(prompts) == 1
    assert "ocean" in prompts[0].prompt and "algorithm" in prompts[0].prompt


def test_associative_score():
    bench = AssociativeBenchmark(AssociativeConfig(pairs=[("ocean", "algorithm")]))
    from cbf.benchmarks.base import RawResults

    raw = RawResults(
        benchmark_id="associative",
        model="test",
        items=make_assoc_result_items(),
        call_ids=[],
        metadata={},
    )
    scores = bench.score(raw)
    assert scores.metrics["chain_validity"] == 0.5
    assert scores.metrics["total_samples"] == 2
    assert "chain_validity" in scores.confidence_intervals


def test_associative_score_empty():
    bench = AssociativeBenchmark()
    from cbf.benchmarks.base import RawResults

    raw = RawResults(benchmark_id="associative", model="x", items=[{"error": "boom"}], call_ids=[])
    scores = bench.score(raw)
    assert scores.metrics["chain_validity"] == 0.0


def test_parse_solution():
    steps = parse_solution("1. Wrap the thread around the ruler\n2. Saw through the bread")
    assert len(steps) == 2
    assert steps[0].startswith("Wrap")


def test_score_solution_feasible_vs_tool_leak():
    task = {"objects": ["a piece of thread", "a metal ruler"], "missing_tool": "a knife"}
    good = score_solution(["Wrap the thread around the ruler tightly", "Saw through the bread with the thread"], task)
    bad = score_solution(["Use a knife", "Cut it"], task)
    assert good["feasibility"] > bad["feasibility"]
    assert bad["tool_leak"] is True
    assert good["tool_leak"] is False


def test_affordance_prompts_and_score():
    bench = AffordanceBenchmark(AffordanceConfig(num_samples=1))
    assert len(bench.prompts()) == 4

    from cbf.benchmarks.base import RawResults

    items = []
    for i, task in enumerate(bench.config.tasks):
        steps = parse_solution(f"1. Use {task['objects'][0]}\n2. Then {task['objects'][1]}")
        items.append({
            "task_idx": i,
            "sample_idx": 0,
            "steps": steps,
            "scores": score_solution(steps, task),
            "valid": True,
            "text": " ".join(steps),
        })
    raw = RawResults(benchmark_id="affordance", model="x", items=items, call_ids=[])
    scores = bench.score(raw)
    assert scores.metrics["feasibility"] > 0
    assert scores.metrics["novelty"] >= 0
    assert scores.metrics["total_samples"] == 4


ENTRY_DICT = {
    "id": "approved-bench",
    "source": "arxiv",
    "title": "Approved Creative Benchmark",
    "url": "https://example.com",
    "summary": "Evaluates divergent association in LLMs.",
    "metadata": {"extracted": {"task_format": "open-ended generation"}},
    "relevance": 0.9,
    "runnability": 0.8,
    "credibility": 0.8,
    "recency": 0.9,
    "score": 0.85,
    "status": "approved",
    "status_reason": "human reviewed",
    "adapter_path": "",
    "first_seen": "2026-10-01T00:00:00+00:00",
    "last_seen": "2026-10-01T00:00:00+00:00",
}


def test_approval_gate_blocks_unapproved():
    entry = RegistryEntry(**ENTRY_DICT)
    entry.status = "candidate"
    with pytest.raises(AdapterApprovalError):
        require_approved(entry)

    entry.status = "approved"
    require_approved(entry)  # no raise


def test_registry_adapter_prompts():
    entry = RegistryEntry(**ENTRY_DICT)
    adapter = RegistryAdapter(entry)
    prompts = adapter.prompts()
    assert len(prompts) >= 1
    assert adapter.id.startswith("adapter_approved-bench")


def test_load_adapter_requires_approval():
    entry = RegistryEntry(**ENTRY_DICT)
    adapter = load_adapter(entry)
    assert adapter.status == "approved"

    entry.status = "adapter_drafted"
    with pytest.raises(AdapterApprovalError):
        load_adapter(entry)


def test_addapter_score_is_real():
    entry = RegistryEntry(**ENTRY_DICT)
    adapter = RegistryAdapter(entry)
    from cbf.benchmarks.base import RawResults

    raw = RawResults(
        benchmark_id=adapter.id,
        model="x",
        items=[
            {"text": "unique alpha", "generation_idx": 0},
            {"text": "unique beta", "generation_idx": 1},
            {"error": "failed", "text": ""},
        ],
        call_ids=[],
        metadata={},
    )
    scores = adapter.score(raw)
    assert scores.metrics["completion_rate"] == pytest.approx(2 / 3)
    assert scores.metrics["n_outputs"] == 2


def test_build_prompts_no_source_fallback():
    entry = RegistryEntry(**ENTRY_DICT)
    entry.metadata = {}
    prompts = build_prompts_from_entry(entry, fallback=["Extra prompt"])
    assert len(prompts) == 2
    assert prompts[1] == "Extra prompt"
