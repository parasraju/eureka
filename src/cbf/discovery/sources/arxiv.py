import arxiv
import asyncio
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timedelta
import json


@dataclass
class ArxivConfig:
    categories: list[str] = field(default_factory=lambda: ["cs.CL", "cs.AI", "cs.HC", "cs.LG"])
    keywords: list[str] = field(default_factory=lambda: [
        "creativity benchmark",
        "divergent thinking",
        "creative writing evaluation",
        "novelty diversity generation",
        "associative creativity",
        "creative problem solving",
        "LLM creativity",
        "language model creativity",
    ])
    max_results: int = 100
    days_back: int = 30


class ArxivSource:
    name = "arxiv"

    def __init__(self, config: ArxivConfig | None = None):
        self.config = config or ArxivConfig()

    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results or self.config.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = []
        for paper in client.results(search):
            results.append({
                "source": "arxiv",
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "published": paper.published.isoformat() if paper.published else None,
                "updated": paper.updated.isoformat() if paper.updated else None,
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "primary_category": paper.primary_category,
                "retrieved_at": datetime.utcnow().isoformat(),
            })
        return results

    async def search_all(self) -> list[dict]:
        all_results = []
        seen_ids = set()

        for keyword in self.config.keywords:
            for category in self.config.categories:
                query = f"cat:{category} AND ({keyword})"
                try:
                    results = await self.search(query)
                    for r in results:
                        if r["arxiv_id"] not in seen_ids:
                            seen_ids.add(r["arxiv_id"])
                            all_results.append(r)
                except Exception as e:
                    print(f"Error searching arXiv for '{query}': {e}")

        return all_results


async def test_arxiv():
    source = ArxivSource(ArxivConfig(
        keywords=["creativity benchmark LLM"],
        max_results=5,
        days_back=7,
    ))
    results = await source.search_all()
    for r in results:
        print(f"- {r['title']} ({r['arxiv_id']})")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    asyncio.run(test_arxiv())