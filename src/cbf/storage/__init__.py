import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import threading


@dataclass
class CallLogEntry:
    run_id: str
    call_id: str
    timestamp_utc: str
    provider: str
    model: str
    request: dict
    response: dict
    latency_s: float
    cost_usd: float
    error: str | None = None


@dataclass
class RunManifest:
    run_id: str
    started_utc: str
    ended_utc: str | None = None
    models: list[str] = field(default_factory=list)
    provider_versions: dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    prompt_set_hashes: dict[str, str] = field(default_factory=dict)
    embedding_model: str = ""
    judge_models: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_calls: int = 0
    status: str = "running"
    error: str | None = None


class BudgetGuard:
    def __init__(self, max_usd: float):
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def check_and_add(self, cost: float) -> bool:
        with self._lock:
            if self.spent_usd + cost > self.max_usd:
                return False
            self.spent_usd += cost
            return True

    def get_spent(self) -> float:
        with self._lock:
            return self.spent_usd

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.max_usd - self.spent_usd)


class CallLogger:
    def __init__(self, run_id: str, runs_dir: str = "runs"):
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)
        self.run_dir = self.runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.calls_file = self.run_dir / "calls.jsonl"
        self.manifest_file = self.run_dir / "manifest.json"
        self._lock = threading.Lock()
        self._manifest = RunManifest(
            run_id=run_id,
            started_utc=datetime.now(timezone.utc).isoformat(),
        )

    def log_call(
        self,
        provider: str,
        model: str,
        request: dict,
        response: dict | None,
        latency_s: float,
        cost_usd: float,
        error: str | None = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        entry = CallLogEntry(
            run_id=self.run_id,
            call_id=call_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            request=request,
            response=response or {},
            latency_s=latency_s,
            cost_usd=cost_usd,
            error=error,
        )

        with self._lock:
            with open(self.calls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

            self._manifest.total_cost_usd += cost_usd
            self._manifest.total_calls += 1

        return call_id

    def update_manifest(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._manifest, key):
                    setattr(self._manifest, key, value)

    def finalize(self, status: str = "completed", error: str | None = None):
        with self._lock:
            self._manifest.ended_utc = datetime.now(timezone.utc).isoformat()
            self._manifest.status = status
            self._manifest.error = error
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self._manifest), f, indent=2, ensure_ascii=False)

    def get_manifest(self) -> RunManifest:
        return self._manifest


def load_manifest(run_id: str, runs_dir: str = "runs") -> RunManifest | None:
    manifest_file = Path(runs_dir) / run_id / "manifest.json"
    if not manifest_file.exists():
        return None
    with open(manifest_file, encoding="utf-8") as f:
        data = json.load(f)
    return RunManifest(**data)


def load_calls(run_id: str, runs_dir: str = "runs") -> list[CallLogEntry]:
    calls_file = Path(runs_dir) / run_id / "calls.jsonl"
    if not calls_file.exists():
        return []
    entries = []
    with open(calls_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(CallLogEntry(**json.loads(line)))
    return entries