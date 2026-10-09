from cbf.judging.panel import WritingEloBenchmark


def create_writing_elo_benchmark(config: dict | None = None) -> WritingEloBenchmark:
    return WritingEloBenchmark(config)