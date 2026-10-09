from dataclasses import dataclass, field
from typing import Protocol, Any
from abc import abstractmethod


@dataclass
class BenchmarkItem:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RawResults:
    benchmark_id: str
    model: str
    items: list[dict]
    call_ids: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class Scores:
    benchmark_id: str
    model: str
    metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class Benchmark(Protocol):
    id: str
    version: str

    def prompts(self) -> list[BenchmarkItem]: ...

    async def run(self, provider, model: str, cfg: dict) -> RawResults: ...

    def score(self, raw: RawResults) -> Scores: ...