from .affordance import AffordanceBenchmark, AffordanceConfig, create_affordance_benchmark
from .associative import AssociativeBenchmark, AssociativeConfig, create_associative_benchmark
from .aut import AUTBenchmark, AUTConfig, create_aut_benchmark
from .base import Benchmark, BenchmarkItem, RawResults, Scores
from .dat import DATBenchmark, DATConfig, create_dat_benchmark
from .diversity import DiversityBenchmark, DiversityConfig, create_diversity_benchmark
from .writing_elo import create_writing_elo_benchmark

BENCHMARK_REGISTRY = {
    "dat": create_dat_benchmark,
    "aut": create_aut_benchmark,
    "diversity": create_diversity_benchmark,
    "writing_elo": create_writing_elo_benchmark,
    "associative": create_associative_benchmark,
    "affordance": create_affordance_benchmark,
}

TIER_A = ["dat", "aut", "diversity"]
TIER_B = ["writing_elo"]
TIER_C = ["associative", "affordance"]


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
    "AssociativeBenchmark",
    "create_associative_benchmark",
    "AssociativeConfig",
    "AffordanceBenchmark",
    "create_affordance_benchmark",
    "AffordanceConfig",
    "BENCHMARK_REGISTRY",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "get_benchmark",
    "list_benchmarks",
]
