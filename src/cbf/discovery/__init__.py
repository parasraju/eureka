from .engine import DiscoveryEngine, DiscoveryConfig, DiscoveredBenchmark, run_discovery
from .sources.arxiv import ArxivSource, ArxivConfig
from .sources.semantic_scholar import SemanticScholarSource, SemanticScholarConfig
from .sources.huggingface import HuggingFaceSource, HuggingFaceConfig
from .sources.github import GitHubSource, GitHubConfig

__all__ = [
    "DiscoveryEngine",
    "DiscoveryConfig",
    "DiscoveredBenchmark",
    "run_discovery",
    "ArxivSource",
    "ArxivConfig",
    "SemanticScholarSource",
    "SemanticScholarConfig",
    "HuggingFaceSource",
    "HuggingFaceConfig",
    "GitHubSource",
    "GitHubConfig",
]