"""Candidate scoring: relevance, runnability, credibility, recency (plan section 3.2)."""

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

CREATIVITY_KEYWORDS = [
    "creativity", "creative", "divergent", "novelty", "originality",
    "imagination", "alternative uses", "associative", "creative writing",
    "story generation", "poetry", "torrance",
]
BENCHMARK_KEYWORDS = [
    "benchmark", "evaluation", "assessment", "metric", "dataset",
    "leaderboard", "challenge", "human evaluation",
]
LLM_KEYWORDS = [
    "llm", "large language model", "language model", "gpt", "claude",
    "gemini", "transformer", "generative ai",
]
PREPRINT_RE = re.compile(r"arxiv|preprint|under review", re.IGNORECASE)
VENUES = ["neurips", "icml", "iclr", "acl", "emnlp", "aaai", "acl findings", "journal"]


@dataclass
class RankedCandidate:
    id: str
    title: str
    url: str
    source: str
    relevance: float = 0.0
    runnability: float = 0.0
    credibility: float = 0.0
    recency: float = 0.0
    score: float = 0.0
    status: str = "candidate"
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def relevance_score(text: str) -> float:
    lower = text.lower()
    score = 0.0
    score += 0.14 * sum(1 for kw in CREATIVITY_KEYWORDS if kw in lower)
    score += 0.10 * sum(1 for kw in BENCHMARK_KEYWORDS if kw in lower)
    score += 0.05 * sum(1 for kw in LLM_KEYWORDS if kw in lower)
    return _clamp(score)


def runnability_score(candidate: dict, code_url: str = "", data_url: str = "", license_name: str = "") -> float:
    score = 0.0
    has_code = bool(code_url) or bool(candidate.get("repo_url")) or bool(candidate.get("html_url"))
    has_data = bool(data_url) or bool(candidate.get("dataset_url"))
    if has_code:
        score += 0.40
    if has_data:
        score += 0.25
    if license_name and license_name.lower() not in ("other", "unknown"):
        score += 0.20
    stars = candidate.get("stars", 0) or 0
    if stars > 50:
        score += 0.15
    elif stars > 10:
        score += 0.08
    return _clamp(score)


def credibility_score(candidate: dict, peer_reviewed: bool | None = None) -> float:
    score = 0.20
    text = " ".join(str(candidate.get(k, "")) for k in ("title", "summary", "abstract", "venue", "published")).lower()
    if peer_reviewed is True:
        score += 0.30
    elif peer_reviewed is False:
        score -= 0.10
    elif PREPRINT_RE.search(text):
        score -= 0.05

    for venue in VENUES:
        if venue in text:
            score += 0.15
            break

    citations = candidate.get("citationCount", 0) or 0
    if citations > 100:
        score += 0.25
    elif citations > 10:
        score += 0.15
    elif citations > 0:
        score += 0.05

    if candidate.get("reproducible") or candidate.get("has_code"):
        score += 0.10
    return _clamp(score)


def recency_score(candidate: dict, now: datetime | None = None) -> float:
    published = candidate.get("published") or candidate.get("year") or candidate.get("published_at") or candidate.get("updated_at")
    if not published:
        return 0.3
    now = now or datetime.now(UTC)
    try:
        if isinstance(published, str):
            pub = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=UTC)
        else:
            pub = datetime(int(published), 1, 1, tzinfo=UTC)
    except (ValueError, TypeError):
        return 0.3
    days_old = max(0, (now - pub).days)
    if days_old < 90:
        return 1.0
    if days_old < 365:
        return 0.8
    if days_old < 365 * 2:
        return 0.5
    return 0.2


def classify(score: float, min_score: float, preprint: bool = False) -> str:
    """Initial status for a candidate. Promotion to 'approved' only happens via human review."""
    if score < min_score:
        return "rejected"
    return "seed" if preprint else "candidate"


def rank_candidate(candidate: dict, min_score: float = 0.3, relevance_floor: float = 0.25) -> RankedCandidate:
    text = " ".join(str(candidate.get(k, "")) for k in ("title", "summary", "abstract", "description", "readme"))

    from cbf.discovery.extractor import extract_rules

    meta = extract_rules(candidate)

    rel = relevance_score(text)
    run = runnability_score(candidate, meta.code_url, meta.data_url, meta.license)
    cred = credibility_score(candidate, meta.peer_reviewed)
    rec = recency_score(candidate)

    score = _clamp(0.40 * rel + 0.25 * run + 0.25 * cred + 0.10 * rec)

    reasons = [f"relevance={rel:.2f}", f"runnability={run:.2f}", f"credibility={cred:.2f}", f"recency={rec:.2f}"]
    if meta.unverified:
        reasons.append("unverified: " + ",".join(meta.unverified))
    if meta.peer_reviewed is False:
        reasons.append("preprint (unreviewed)")

    if rel < relevance_floor:
        status = "rejected"
        reasons.append(f"relevance {rel:.2f} below floor {relevance_floor}")
        score = _clamp(score * 0.25)
    else:
        status = classify(score, min_score, preprint=meta.peer_reviewed is False)
        if score < min_score:
            reasons.append(f"below min_score {min_score}")

    candidate_id = candidate.get("id") or candidate.get("arxiv_id") or candidate.get("paperId") or meta.name
    return RankedCandidate(
        id=str(candidate_id),
        title=candidate.get("title") or candidate.get("name") or candidate.get("full_name") or "Untitled",
        url=meta.source_url or candidate.get("url", ""),
        source=candidate.get("source", "unknown"),
        relevance=round(rel, 4),
        runnability=round(run, 4),
        credibility=round(cred, 4),
        recency=round(rec, 4),
        score=round(score, 4),
        status=status,
        reasons=reasons,
        metadata={"extracted": meta.to_dict(), "raw": {k: candidate.get(k) for k in ("stars", "citationCount", "published")}},
    )


def rank(candidates: list[dict], min_score: float = 0.3, relevance_floor: float = 0.25) -> list[RankedCandidate]:
    ranked = []
    for c in candidates:
        data = c.to_dict() if hasattr(c, "to_dict") else dict(c)
        ranked.append(rank_candidate(data, min_score, relevance_floor=relevance_floor))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
