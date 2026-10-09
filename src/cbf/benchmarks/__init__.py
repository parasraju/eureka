from .base import Benchmark, BenchmarkItem, RawResults, Scores
from .dat import DATBenchmark, create_dat_benchmark, DATConfig
from .aut import AUTBenchmark, create_aut_benchmark, AUTConfig
from .diversity import DiversityBenchmark, create_diversity_benchmark, DiversityConfig
from .writing_elo import create_writing_elo_benchmark

BENCHMARK_REGISTRY = {
    "dat": create_dat_benchmark,
    "aut": create_aut_benchmark,
    "diversity": create_diversity_benchmark,
    "writing_elo": create_writing_elo_benchmark,
}

TIER_A = ["dat", "aut", "diversity"]
TIER_B = ["writing_elo"]
TIER_C = []


def get_benchmark(name: str, config: dict | None = None) -> Benchmark:
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    return BENCHMARK_REGISTRY[name](config)


def list_benchmarks() -> list[str]:
    return list(BENCHMARK_REGISTRY.keys())


__all__ = [
    "Benchmark",
    "BenchmarkItem",
    "RawResults",
    "Scores",
    "DATBenchmark",
    "create_dat_benchmark",
    "DATConfig",
    "AUTBenchmark",
    "create_aut_benchmark",
    "AUTConfig",
    "DiversityBenchmark",
    "create_diversity_benchmark",
    "DiversityConfig",
    "BENCHMARK_REGISTRY",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "get_benchmark",
    "list_benchmarks",
]