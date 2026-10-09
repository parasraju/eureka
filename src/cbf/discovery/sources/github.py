import asyncio
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
from github import Github
import os


@dataclass
class GitHubConfig:
    token: str | None = None
    queries: list[str] = field(default_factory=lambda: [
        "creativity benchmark language model",
        "creative writing benchmark",
        "divergent thinking LLM",
        "novelty benchmark LLM",
        "creative evaluation language model",
    ])
    limit: int = 30


class GitHubSource:
    name = "github"

    def __init__(self, config: GitHubConfig | None = None):
        self.config = config or GitHubConfig()
        self.token = self.config.token or os.getenv("GITHUB_TOKEN")
        self.gh = Github(self.token) if self.token else Github()

    async def search_repos(self) -> list[dict]:
        all_results = []
        for query in self.config.queries:
            try:
                repos = self.gh.search_repositories(query=query, sort="stars", order="desc")
                count = 0
                for repo in repos:
                    if count >= self.config.limit:
                        break
                    all_results.append({
                        "source": "github",
                        "type": "repository",
                        "id": repo.full_name,
                        "name": repo.name,
                        "full_name": repo.full_name,
                        "description": repo.description,
                        "stars": repo.stargazers_count,
                        "forks": repo.forks_count,
                        "language": repo.language,
                        "topics": repo.get_topics(),
                        "created_at": repo.created_at.isoformat() if repo.created_at else None,
                        "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                        "url": repo.html_url,
                        "license": repo.license.name if repo.license else None,
                        "retrieved_at": datetime.utcnow().isoformat(),
                    })
                    count += 1
            except Exception as e:
                print(f"Error searching GitHub for '{query}': {e}")
        return all_results

    async def get_readme(self, repo_full_name: str) -> str | None:
        try:
            repo = self.gh.get_repo(repo_full_name)
            readme = repo.get_readme()
            return readme.decoded_content.decode("utf-8")
        except Exception:
            return None

    async def search_all(self) -> list[dict]:
        return await self.search_repos()


async def test_github():
    source = GitHubSource(GitHubConfig(
        queries=["creativity benchmark language model"],
        limit=5,
    ))
    results = await source.search_all()
    for r in results:
        print(f"- {r['full_name']}: {r['description'][:80] if r['description'] else 'No description'} (★{r['stars']})")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    asyncio.run(test_github())