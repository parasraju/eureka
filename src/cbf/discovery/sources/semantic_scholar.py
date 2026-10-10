import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime

import httpx


@dataclass
class SemanticScholarConfig:
    api_key: str | None = None
    keywords: list[str] = field(default_factory=lambda: [
        "creativity benchmark",
        "divergent thinking language model",
        "creative writing evaluation",
        "novelty diversity generation",
        "associative creativity LLM",
        "creative problem solving benchmark",
    ])
    limit: int = 100
    fields: list[str] = field(default_factory=lambda: [
        "paperId", "title", "abstract", "authors", "year", "venue",
        "citationCount", "influentialCitationCount", "url", "doi",
        "references", "citations", "fieldsOfStudy", "publicationTypes"
    ])


class SemanticScholarSource:
    name = "semantic_scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, config: SemanticScholarConfig | None = None):
        self.config = config or SemanticScholarConfig()
        self.api_key = self.config.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str, limit: int | None = None) -> list[dict]:
        params = {
            "query": query,
            "limit": limit or self.config.limit,
            "fields": ",".join(self.config.fields),
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await self.client.get(f"{self.BASE_URL}/paper/search", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"Error searching Semantic Scholar for '{query}': {e}")
            return []

    async def get_paper_details(self, paper_id: str) -> dict | None:
        params = {"fields": ",".join(self.config.fields)}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await self.client.get(f"{self.BASE_URL}/paper/{paper_id}", params=params, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting paper {paper_id}: {e}")
            return None

    async def get_citing_papers(self, paper_id: str, limit: int = 50) -> list[dict]:
        params = {"limit": limit, "fields": ",".join(self.config.fields)}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            response = await self.client.get(f"{self.BASE_URL}/paper/{paper_id}/citations", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"Error getting citations for {paper_id}: {e}")
            return []

    async def search_all(self) -> list[dict]:
        all_results = []
        seen_ids = set()

        for keyword in self.config.keywords:
            try:
                results = await self.search(keyword)
                for r in results:
                    pid = r.get("paperId")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        r["source"] = "semantic_scholar"
                        r["retrieved_at"] = datetime.utcnow().isoformat()
                        all_results.append(r)
            except Exception as e:
                print(f"Error searching Semantic Scholar for '{keyword}': {e}")

        return all_results

    async def close(self):
        await self.client.aclose()


async def test_semantic_scholar():
    source = SemanticScholarSource(SemanticScholarConfig(
        keywords=["creativity benchmark LLM"],
        limit=5,
    ))
    results = await source.search_all()
    for r in results:
        print(f"- {r.get('title', 'No title')} ({r.get('paperId', 'No ID')})")
    print(f"Total: {len(results)}")
    await source.close()


if __name__ == "__main__":
    asyncio.run(test_semantic_scholar())
