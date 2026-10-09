import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from .sources.arxiv import ArxivSource, ArxivConfig
from .sources.semantic_scholar import SemanticScholarSource, SemanticScholarConfig
from .sources.huggingface import HuggingFaceSource, HuggingFaceConfig
from .sources.github import GitHubSource, GitHubConfig


@dataclass
class DiscoveryConfig:
    sources: list[str] = field(default_factory=lambda: ["arxiv", "semantic_scholar", "huggingface", "github"])
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    semantic_scholar: SemanticScholarConfig = field(default_factory=SemanticScholarConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    output_dir: str = "data/discovery"
    min_score: float = 0.5


@dataclass
class DiscoveredBenchmark:
    id: str
    source: str
    title: str
    url: str
    summary: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    status: str = "candidate"  # candidate, approved, rejected
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DiscoveryEngine:
    def __init__(self, config: DiscoveryConfig | None = None):
        self.config = config or DiscoveryConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sources = self._init_sources()

    def _init_sources(self) -> dict:
        sources = {}
        if "arxiv" in self.config.sources:
            sources["arxiv"] = ArxivSource(self.config.arxiv)
        if "semantic_scholar" in self.config.sources:
            sources["semantic_scholar"] = SemanticScholarSource(self.config.semantic_scholar)
        if "huggingface" in self.config.sources:
            sources["huggingface"] = HuggingFaceSource(self.config.huggingface)
        if "github" in self.config.sources:
            sources["github"] = GitHubSource(self.config.github)
        return sources

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for r in results:
            key = None
            if r.get("arxiv_id"):
                key = f"arxiv:{r['arxiv_id']}"
            elif r.get("paperId"):
                key = f"semantic_scholar:{r['paperId']}"
            elif r.get("id") and r.get("source") == "huggingface":
                key = f"huggingface:{r['id']}"
            elif r.get("id") and r.get("source") == "github":
                key = f"github:{r['id']}"

            if key and key not in seen:
                seen.add(key)
                unique.append(r)
            elif not key:
                unique.append(r)
        return unique

    def _score_candidate(self, candidate: dict) -> float:
        score = 0.0

        text = " ".join([
            candidate.get("title", ""),
            candidate.get("summary", ""),
            candidate.get("description", ""),
        ]).lower()

        creativity_keywords = [
            "creativity", "creative", "divergent", "novelty", "originality",
            "imagination", "innovation", "alternative uses", "associative",
            "creative writing", "story generation", "poetry"
        ]
        benchmark_keywords = [
            "benchmark", "evaluation", "assessment", "metric", "dataset",
            "leaderboard", "competition", "challenge"
        ]
        llm_keywords = [
            "llm", "large language model", "language model", "gpt", "claude",
            "gemini", "transformer", "generative ai"
        ]

        for kw in creativity_keywords:
            if kw in text:
                score += 0.15
        for kw in benchmark_keywords:
            if kw in text:
                score += 0.1
        for kw in llm_keywords:
            if kw in text:
                score += 0.05

        if candidate.get("citationCount", 0) > 10:
            score += 0.1
        if candidate.get("citationCount", 0) > 100:
            score += 0.1
        if candidate.get("stars", 0) > 50:
            score += 0.1
        if candidate.get("stars", 0) > 500:
            score += 0.1

        published = candidate.get("published") or candidate.get("year") or candidate.get("published_at")
        if published:
            try:
                if isinstance(published, str):
                    pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                else:
                    pub_date = datetime(published, 1, 1)
                days_old = (datetime.utcnow() - pub_date).days
                if days_old < 365:
                    score += 0.1
                if days_old < 90:
                    score += 0.1
            except Exception:
                pass

        return min(score, 1.0)

    async def discover(self) -> list[DiscoveredBenchmark]:
        all_results = []

        for name, source in self.sources.items():
            print(f"Searching {name}...")
            try:
                results = await source.search_all()
                print(f"  Found {len(results)} results")
                all_results.extend(results)
            except Exception as e:
                print(f"Error searching {name}: {e}")

        unique_results = self._deduplicate(all_results)
        print(f"After deduplication: {len(unique_results)} unique results")

        benchmarks = []
        for r in unique_results:
            score = self._score_candidate(r)
            if score < self.config.min_score:
                continue

            benchmark_id = self._generate_id(r)
            benchmark = DiscoveredBenchmark(
                id=benchmark_id,
                source=r.get("source", "unknown"),
                title=r.get("title") or r.get("name") or r.get("full_name") or "Untitled",
                url=r.get("url") or r.get("pdf_url") or r.get("html_url") or "",
                summary=r.get("summary") or r.get("description") or r.get("abstract") or "",
                metadata=r,
                score=score,
            )
            benchmarks.append(benchmark)

        benchmarks.sort(key=lambda x: x.score, reverse=True)
        return benchmarks

    def _generate_id(self, result: dict) -> str:
        content = json.dumps(result, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def save_results(self, benchmarks: list[DiscoveredBenchmark], filename: str | None = None):
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"discovery_{timestamp}.json"

        filepath = self.output_dir / filename
        data = [asdict(b) for b in benchmarks]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(benchmarks)} benchmarks to {filepath}")
        return filepath

    def load_results(self, filename: str) -> list[DiscoveredBenchmark]:
        filepath = self.output_dir / filename
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return [DiscoveredBenchmark(**d) for d in data]


async def run_discovery(config: DiscoveryConfig | None = None) -> list[DiscoveredBenchmark]:
    engine = DiscoveryEngine(config)
    benchmarks = await engine.discover()
    engine.save_results(benchmarks)
    return benchmarks


if __name__ == "__main__":
    config = DiscoveryConfig(
        arxiv=ArxivConfig(keywords=["creativity benchmark LLM"], max_results=10),
        semantic_scholar=SemanticScholarConfig(keywords=["creativity benchmark LLM"], limit=10),
        huggingface=HuggingFaceConfig(dataset_tags=["creativity"], limit=5),
        github=GitHubConfig(queries=["creativity benchmark language model"], limit=5),
        min_score=0.3,
    )
    results = asyncio.run(run_discovery(config))
    print(f"\nTop {min(10, len(results))} results:")
    for i, b in enumerate(results[:10]):
        print(f"  {i+1}. [{b.source}] {b.title[:80]} (score: {b.score:.2f})")