"""Auto-generated wrappers for newly discovered, human-approved benchmarks.

Every adapter implements the Benchmark interface from a registry entry's metadata.
The approval gate (require_approved) is mandatory: unapproved entries raise
AdapterApprovalError before any API call is made, so a wrong auto-generated
implementation can never reach a published leaderboard.
"""

from cbf.discovery.ranker import RankedCandidate
from cbf.discovery.registry_db import RegistryEntry

from ..base import BenchmarkItem, RawResults, Scores


class AdapterApprovalError(RuntimeError):
    """Raised when an adapter is used without human approval."""


class AdapterNotRunnable(RuntimeError):
    """Raised when a registry entry has no real prompts to run."""


APPROVAL_STATUSES = {"approved"}


def require_approved(entry: RegistryEntry | RankedCandidate | dict, allow_status: set[str] | None = None) -> None:
    allowed = allow_status or APPROVAL_STATUSES
    status = entry.get("status") if isinstance(entry, dict) else getattr(entry, "status", "")
    entry_id = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", "?")
    if status not in allowed:
        raise AdapterApprovalError(
            f"Benchmark {entry_id!r} has status {status!r}; human approval required "
            f"(one of {sorted(allowed)}) before it can run on a published leaderboard. "
            f"Run: cbf registry approve {entry_id}"
        )


def build_prompts_from_entry(entry: RegistryEntry | dict, fallback: list[str] | None = None) -> list[str]:
    """Return the entry's stored prompt list, or refuse to fabricate one."""
    data = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
    metadata = data.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = {}
    raw = metadata.get("raw", {}) if isinstance(metadata, dict) else {}

    prompts = raw.get("prompts") if isinstance(raw, dict) else None
    if isinstance(prompts, list) and prompts:
        return [str(p) for p in prompts]

    entry_id = data.get("id", "?")
    raise AdapterNotRunnable(
        f"Registry entry {entry_id} has no runnable prompts; extract them from the "
        f"source paper/repo and store them before running"
    )


class RegistryAdapter:
    """Generic adapter: runs a discovered benchmark's prompt set through the standard runner."""

    def __init__(self, entry: RegistryEntry | dict, version: str = "0.0.0-draft"):
        if isinstance(entry, dict):
            entry = RegistryEntry(**{k: entry.get(k) for k in RegistryEntry.__dataclass_fields__})
        self.entry = entry
        self.id = f"adapter_{entry.id}"
        self.version = version
        self.prompts_list = build_prompts_from_entry(entry)

    @property
    def status(self) -> str:
        return self.entry.status

    def assert_approved(self) -> None:
        require_approved(self.entry)

    def prompts(self) -> list[BenchmarkItem]:
        return [
            BenchmarkItem(id=f"{self.id}_{i}", prompt=p, metadata={"prompt_idx": i})
            for i, p in enumerate(self.prompts_list)
        ]

    async def run(self, provider, model: str, cfg: dict) -> RawResults:
        self.assert_approved()
        import asyncio

        from cbf.providers import LLMRequest

        k = cfg.get("k_generations", 1)
        temperature = cfg.get("temperature", 0.7)
        semaphore = asyncio.Semaphore(cfg.get("concurrency", 5))
        call_ids: list[str] = []
        items: list[dict] = []

        async def gen(item: BenchmarkItem, idx: int):
            async with semaphore:
                req = LLMRequest(
                    model=model,
                    messages=[{"role": "user", "content": item.prompt}],
                    temperature=temperature,
                    max_tokens=cfg.get("max_tokens", 500),
                    seed=cfg.get("seed"),
                )
                resp = await provider.complete(req)
                call_ids.append(resp.request_id)
                return {
                    "prompt_id": item.id,
                    "prompt_idx": item.metadata["prompt_idx"],
                    "generation_idx": idx,
                    "text": resp.text,
                    "call_id": resp.request_id,
                    "latency_s": resp.latency_s,
                    "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
                }

        tasks = [gen(item, i) for item in self.prompts() for i in range(k)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                items.append({"error": str(r), "text": ""})
            else:
                items.append(r)

        return RawResults(
            benchmark_id=self.id,
            model=model,
            items=items,
            call_ids=call_ids,
            metadata={"adapter_of": self.entry.id, "status": self.entry.status, "config": cfg},
        )

    def score(self, raw: RawResults) -> Scores:
        ok = [i for i in raw.items if "error" not in i and i.get("text")]
        from cbf.metrics import bootstrap_ci, distinct_n

        texts = [i["text"] for i in ok]
        return Scores(
            benchmark_id=self.id,
            model=raw.model,
            metrics={
                "completion_rate": len(ok) / len(raw.items) if raw.items else 0.0,
                "distinct_1": distinct_n(texts, 1) if texts else 0.0,
                "distinct_2": distinct_n(texts, 2) if texts else 0.0,
                "n_outputs": len(ok),
            },
            confidence_intervals={
                "completion_rate": bootstrap_ci([1.0 if "error" not in i and i.get("text") else 0.0 for i in raw.items]),
            },
            metadata={
                "adapter_of": self.entry.id,
                "requires_approval": True,
                "proxy_metrics_only": True,
            },
        )


def load_adapter(entry: RegistryEntry | dict, allow_status: set[str] | None = None) -> RegistryAdapter:
    """Load an adapter, refusing entry statuses that are not approved."""
    require_approved(entry, allow_status=allow_status or {"approved"})
    return RegistryAdapter(entry)
