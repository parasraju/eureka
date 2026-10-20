"""SQLite registry of discovered benchmarks with status workflow and digests.

Status workflow (plan section 3.2):
    seed -> candidate -> adapter_drafted -> approved -> rejected(reason)
Any non-approved status may move to rejected. Rejecting an approved entry
requires a non-empty reason. Illegal transitions raise ValueError.
Only 'approved' entries may run on a published leaderboard (human approval gate).
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

VALID_STATUSES = ("seed", "candidate", "adapter_drafted", "approved", "rejected")

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "seed": {"candidate", "rejected"},
    "candidate": {"adapter_drafted", "rejected"},
    "adapter_drafted": {"approved", "rejected"},
    "approved": {"rejected"},
    "rejected": set(),
}


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'unknown',
    title TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    relevance REAL NOT NULL DEFAULT 0,
    runnability REAL NOT NULL DEFAULT 0,
    credibility REAL NOT NULL DEFAULT 0,
    recency REAL NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'candidate',
    status_reason TEXT NOT NULL DEFAULT '',
    adapter_path TEXT NOT NULL DEFAULT '',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_benchmarks_status ON benchmarks(status);
CREATE INDEX IF NOT EXISTS idx_benchmarks_score ON benchmarks(score DESC);
"""


@dataclass
class RegistryEntry:
    id: str
    source: str
    title: str
    url: str
    summary: str
    metadata: dict
    relevance: float
    runnability: float
    credibility: float
    recency: float
    score: float
    status: str
    status_reason: str
    adapter_path: str
    first_seen: str
    last_seen: str

    def to_dict(self) -> dict:
        return asdict(self)


class RegistryDB:
    def __init__(self, path: str = "data/registry.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def upsert(self, entries: list) -> int:
        """Insert or update ranked candidates (RankedCandidate or dict). Preserves status on re-discovery."""
        now = datetime.now(UTC).isoformat()
        count = 0
        for entry in entries:
            data = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
            existing = self.conn.execute("SELECT id, status, status_reason, first_seen FROM benchmarks WHERE id = ?", (data["id"],)).fetchone()
            metadata = data.get("metadata", {})
            if not isinstance(metadata, str):
                metadata = json.dumps(metadata, ensure_ascii=False)

            if existing:
                self.conn.execute(
                    """UPDATE benchmarks SET source=?, title=?, url=?, summary=?, metadata=?,
                       relevance=?, runnability=?, credibility=?, recency=?, score=?, last_seen=?
                       WHERE id=?""",
                    (
                        data.get("source", "unknown"),
                        data.get("title", ""),
                        data.get("url", ""),
                        data.get("summary", ""),
                        metadata,
                        data.get("relevance", 0),
                        data.get("runnability", 0),
                        data.get("credibility", 0),
                        data.get("recency", 0),
                        data.get("score", 0),
                        now,
                        data["id"],
                    ),
                )
            else:
                status = data.get("status", "candidate")
                if status not in VALID_STATUSES:
                    status = "candidate"
                self.conn.execute(
                    """INSERT INTO benchmarks (id, source, title, url, summary, metadata,
                       relevance, runnability, credibility, recency, score, status, status_reason,
                       adapter_path, first_seen, last_seen)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        data["id"],
                        data.get("source", "unknown"),
                        data.get("title", ""),
                        data.get("url", ""),
                        data.get("summary", ""),
                        metadata,
                        data.get("relevance", 0),
                        data.get("runnability", 0),
                        data.get("credibility", 0),
                        data.get("recency", 0),
                        data.get("score", 0),
                        status,
                        data.get("status_reason", "") or " ".join(data.get("reasons", [])),
                        data.get("adapter_path", ""),
                        now,
                        now,
                    ),
                )
            count += 1
        self.conn.commit()
        return count

    def get(self, entry_id: str) -> RegistryEntry | None:
        row = self.conn.execute("SELECT * FROM benchmarks WHERE id = ?", (entry_id,)).fetchone()
        return self._to_entry(row) if row else None

    def search(self, status: str | None = None, min_score: float = 0.0) -> list[RegistryEntry]:
        query = "SELECT * FROM benchmarks WHERE score >= ?"
        params: list = [min_score]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY score DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._to_entry(r) for r in rows]

    def set_status(self, entry_id: str, status: str, reason: str = "") -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}. Valid: {VALID_STATUSES}")

        row = self.conn.execute(
            "SELECT status FROM benchmarks WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No benchmark with id {entry_id!r}")

        current = row["status"]
        if status != current and status not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"Illegal status transition {current!r} -> {status!r} for {entry_id!r}"
            )
        if current == "approved" and status == "rejected" and not reason.strip():
            raise ValueError("Rejecting an approved benchmark requires a non-empty reason")

        self.conn.execute(
            "UPDATE benchmarks SET status = ?, status_reason = ? WHERE id = ?",
            (status, reason, entry_id),
        )
        self.conn.commit()

    def set_adapter(self, entry_id: str, adapter_path: str) -> None:
        self.conn.execute(
            "UPDATE benchmarks SET adapter_path = ?, status = 'adapter_drafted' WHERE id = ?",
            (adapter_path, entry_id),
        )
        self.conn.commit()

    def approved(self) -> list[RegistryEntry]:
        return self.search(status="approved")

    def new_since(self, days: int) -> list[RegistryEntry]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        rows = self.conn.execute(
            "SELECT * FROM benchmarks ORDER BY score DESC"
        ).fetchall()
        entries = [self._to_entry(r) for r in rows]
        return [entry for entry in entries if _parse_iso(entry.first_seen) >= cutoff]

    def digest(self, days: int = 7) -> str:
        """Markdown 'what's new' digest for the last N days."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        entries = self.new_since(days)
        approved = [e for e in self.approved() if _parse_iso(e.first_seen) >= cutoff]
        lines = [
            f"# Benchmark discovery digest (last {days} days)",
            "",
            f"*Generated {datetime.now(UTC).isoformat()}*",
            "",
            f"New candidates: **{len(entries)}** · Newly approved: **{len(approved)}**",
            "",
        ]
        if not entries:
            lines.append("_No new benchmarks found in this window._")
            return "\n".join(lines)

        lines.append("| Score | Status | Title | Source | URL |")
        lines.append("|------:|--------|-------|--------|-----|")
        for e in entries[:50]:
            title = e.title.replace("|", "/")
            lines.append(f"| {e.score:.2f} | {e.status} | {title} | {e.source} | {e.url} |")

        if len(entries) > 50:
            lines.append("")
            lines.append(f"_... and {len(entries) - 50} more_")

        flagged = [e for e in entries if "preprint" in e.status_reason]
        if flagged:
            lines += ["", "## Needs verification (preprints)", ""]
            for e in flagged[:10]:
                lines.append(f"- [{e.title}]({e.url}) — {e.status_reason}")

        return "\n".join(lines)

    def _to_entry(self, row: sqlite3.Row) -> RegistryEntry:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (TypeError, json.JSONDecodeError):
            d["metadata"] = {}
        return RegistryEntry(**d)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.conn.close()
