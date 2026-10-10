"""Metadata extraction from discovered papers/repositories.

Rule-based extraction always runs and quotes where each fact came from.
Optional LLM-assisted extraction fills gaps, but every extracted value is
re-checked against the source text; unverifiable fields are marked unverified.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass
class ExtractedMetadata:
    name: str = ""
    measures: str = ""
    task_format: str = ""
    metric: str = ""
    judge: str = ""
    dataset_size: str = ""
    license: str = ""
    code_url: str = ""
    data_url: str = ""
    date: str = ""
    peer_reviewed: bool | None = None
    evidence: dict[str, str] = field(default_factory=dict)   # field -> quoted source location/text
    unverified: list[str] = field(default_factory=list)
    source_url: str = ""
    retrieval_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_LICENSE_RE = re.compile(
    r"\b(MIT|Apache(?:-2\.0)?|GPL(?:-[23]\.0)?|BSD-[23]-Clause|CC BY(?:-SA)?(?:-4\.0)?|"
    r"CC0|Llama|OpenRAIL|other)\b",
    re.IGNORECASE,
)
_METRIC_RE = re.compile(
    r"\b(BLEU|ROUGE|Self-BLEU|distinct-n|Elo|Bradley-Terry|accuracy|F1|human evaluation|"
    r"LLM[- ]judge|rubric|Krippendorff|agreement|percentile|embedding distance)\b",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(r"([\d,]+)\s+(?:prompts|items|samples|questions|instances|examples|generations)", re.IGNORECASE)
_VENUE_RE = re.compile(r"\b(aaaai?|neurips|icml|iclr|acl|emnlp|eacl|naacl|findings|journal|arxiv|preprint|workshop)\b", re.IGNORECASE)


def _quote(text: str, match: re.Match, width: int = 160) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    return text[start:end].replace("\n", " ").strip()


def extract_rules(candidate: dict) -> ExtractedMetadata:
    title = candidate.get("title") or candidate.get("name") or candidate.get("full_name") or ""
    body = " ".join(
        str(candidate.get(k, "")) for k in ("summary", "abstract", "description", "readme", "text")
    )
    combined = f"{title}\n{body}"
    lower = combined.lower()

    meta = ExtractedMetadata(name=title.strip())

    url = candidate.get("url") or candidate.get("pdf_url") or candidate.get("html_url") or ""
    meta.source_url = url

    meta.code_url = candidate.get("repo_url") or candidate.get("html_url") or ""
    for m in _URL_RE.finditer(body):
        u = m.group(0)
        if not meta.code_url and any(h in u for h in ("github.com", "gitlab.com", "huggingface.co")):
            meta.code_url = u
        if not meta.data_url and any(h in u for h in ("huggingface.co/datasets", "zenodo.org", "doi.org/10.5281")):
            meta.data_url = u

    lic = _LICENSE_RE.search(combined)
    if lic:
        meta.license = lic.group(0)
        meta.evidence["license"] = _quote(combined, lic)
    elif candidate.get("license"):
        meta.license = str(candidate["license"])
        meta.evidence["license"] = f"source field: license={candidate['license']}"
    else:
        meta.unverified.append("license")

    metric = _METRIC_RE.search(combined)
    if metric:
        meta.metric = metric.group(0)
        meta.evidence["metric"] = _quote(combined, metric)
    else:
        meta.unverified.append("metric")

    if "llm-judge" in lower or "llm judge" in lower or "gpt-4" in lower or "as a judge" in lower:
        meta.judge = "LLM judge (see evidence)"
        meta.evidence["judge"] = "phrase found in source text"
    elif "human evaluation" in lower or "human evaluators" in lower:
        meta.judge = "human evaluators"
        meta.evidence["judge"] = "phrase found in source text"

    size = _SIZE_RE.search(combined)
    if size:
        meta.dataset_size = size.group(0)
        meta.evidence["dataset_size"] = _quote(combined, size)

    venue = _VENUE_RE.search(combined)
    if venue:
        v = venue.group(0).lower()
        if v in ("arxiv", "preprint"):
            meta.peer_reviewed = False
        elif v == "findings":
            meta.peer_reviewed = True
        meta.evidence["venue"] = _quote(combined, venue)

    published = candidate.get("published") or candidate.get("year") or candidate.get("published_at")
    if published:
        meta.date = str(published)

    if "benchmark" in lower or "we evaluate" in lower or "we propose an evaluation" in lower:
        meta.task_format = "evaluation benchmark"
        meta.evidence["task_format"] = "keyword match in title/abstract"
        meta.measures = "creativity evaluation of language models" if "creativ" in lower else ""
    if not meta.measures and "creativ" in lower:
        meta.measures = "creativity-related capability"
        meta.evidence["measures"] = "keyword 'creativ*' found in source"

    if meta.peer_reviewed is None:
        meta.unverified.append("peer_reviewed")

    return meta


_EXTRACT_PROMPT = """Extract benchmark metadata from the source below. Respond with ONLY a JSON object:
{{"name": str, "measures": str, "task_format": str, "metric": str, "judge": str,
  "dataset_size": str, "license": str, "code_url": str, "data_url": str,
  "peer_reviewed": bool | null}}
Only include facts stated in the source. Use null/empty string when not stated.

SOURCE URL: {url}
SOURCE TEXT:
{text}
"""


async def extract_with_llm(candidate: dict, provider, model: str) -> ExtractedMetadata:
    """LLM-assisted extraction, rule-checked: values absent from the source are marked unverified."""
    base = extract_rules(candidate)
    text = " ".join(str(candidate.get(k, "")) for k in ("title", "summary", "abstract", "description", "readme"))
    from cbf.providers import LLMRequest

    req = LLMRequest(
        model=model,
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(url=base.source_url, text=text[:8000])}],
        temperature=0.0,
        max_tokens=700,
    )
    resp = await provider.complete(req)
    try:
        start = resp.text.find("{")
        end = resp.text.rfind("}")
        data = json.loads(resp.text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        base.unverified.append("llm_extraction:unparseable")
        return base

    lower_text = text.lower()
    for key, value in data.items():
        if value in (None, "", "null"):
            continue
        if isinstance(value, str) and len(value) > 3 and not isinstance(value, bool):
            if value.lower() not in lower_text and key in ("license", "metric", "dataset_size", "code_url", "data_url"):
                base.unverified.append(key)
                continue
        if key == "peer_reviewed":
            base.peer_reviewed = value if isinstance(value, bool) else None
            continue
        if hasattr(base, key) and not getattr(base, key):
            setattr(base, key, value)
            base.evidence[key] = "LLM extraction, checked against source text"

    return base


def extract(candidate: dict) -> ExtractedMetadata:
    """Synchronous rule-based extraction (no API calls)."""
    return extract_rules(candidate)
