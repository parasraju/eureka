import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime

from huggingface_hub import HfApi, list_datasets, list_papers


@dataclass
class HuggingFaceConfig:
    token: str | None = None
    dataset_tags: list[str] = field(default_factory=lambda: [
        "creativity", "creative-writing", "divergent-thinking",
        "creative-evaluation", "novelty", "creative-problem-solving"
    ])
    paper_keywords: list[str] = field(default_factory=lambda: [
        "creativity benchmark", "divergent thinking", "creative writing"
    ])
    limit: int = 50


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, config: HuggingFaceConfig | None = None):
        self.config = config or HuggingFaceConfig()
        self.token = self.config.token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def _serialize_value(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, '__dict__'):
            return str(value)
        return value

    async def search_datasets(self) -> list[dict]:
        all_results = []
        for tag in self.config.dataset_tags:
            try:
                datasets = list_datasets(filter=tag, limit=self.config.limit, token=self.token)
                for ds in datasets:
                    all_results.append({
                        "source": "huggingface",
                        "type": "dataset",
                        "id": ds.id,
                        "title": getattr(ds, "title", ds.id),
                        "description": getattr(ds, "description", ""),
                        "tags": getattr(ds, "tags", []),
                        "downloads": getattr(ds, "downloads", 0),
                        "likes": getattr(ds, "likes", 0),
                        "last_modified": self._serialize_value(getattr(ds, "lastModified", None)),
                        "url": f"https://huggingface.co/datasets/{ds.id}",
                        "retrieved_at": datetime.utcnow().isoformat(),
                    })
            except Exception as e:
                print(f"Error searching HF datasets for tag '{tag}': {e}")
        return all_results

    async def search_papers(self) -> list[dict]:
        all_results = []
        for keyword in self.config.paper_keywords:
            try:
                papers = list_papers(query=keyword, limit=self.config.limit, token=self.token)
                for paper in papers:
                    paper_id = getattr(paper, 'paper_id', None) or getattr(paper, 'id', None) or paper.title[:50]
                    all_results.append({
                        "source": "huggingface",
                        "type": "paper",
                        "id": paper_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "summary": paper.summary,
                        "published_at": self._serialize_value(getattr(paper, 'published_at', None)),
                        "url": paper.url,
                        "upvotes": getattr(paper, 'upvotes', 0),
                        "retrieved_at": datetime.utcnow().isoformat(),
                    })
            except Exception as e:
                print(f"Error searching HF papers for '{keyword}': {e}")
        return all_results

    async def search_all(self) -> list[dict]:
        datasets = await self.search_datasets()
        papers = await self.search_papers()
        return datasets + papers


async def test_huggingface():
    source = HuggingFaceSource(HuggingFaceConfig(
        dataset_tags=["creativity"],
        paper_keywords=["creativity benchmark"],
        limit=5,
    ))
    results = await source.search_all()
    for r in results:
        print(f"- [{r['type']}] {r['title']} ({r['id']})")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    asyncio.run(test_huggingface())
