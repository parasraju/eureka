"""Metadata extraction from discovered papers/repositories.

Rule-based extraction always runs and records character offsets into the
combined source text for every fact. Optional LLM-assisted extraction fills
gaps, but every extracted string is re-checked against the source text:
a normalized substring match or >= 80% token overlap (for values longer than
3 characters) is required. Values that fail verification are listed in
``unverified`` and never written to the metadata.

Rule-inferred fields (``task_format`` and ``measures``) are recorded in
``derived``; they are not treated as extracted facts. Missing metadata is
never fabricated.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

PREPRINT_RE = re.compile(r"\barxiv\b|\bpreprint\b|\bunder review\b", re.IGNORECASE)
VENUES = ("neurips", "icml", "iclr", "acl", "emnlp", "aaai", "findings", "journal")
VENUE_FIELD_KEYS = (
    "venue", "publication_venue", "publicationVenue", "journal", "conference", "booktitle",
)


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
    evidence: dict[str, dict] = field(default_factory=dict)  # field -> {quote, start, end}
    derived: list[str] = field(default_factory=list)
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
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_JUDGE_PATTERNS = (
    re.compile(r"llm[- ]judge", re.IGNORECASE),
    re.compile(r"gpt-4(?:o)?", re.IGNORECASE),
    re.compile(r"as a judge", re.IGNORECASE),
    re.compile(r"human evaluation", re.IGNORECASE),
    re.compile(r"human evaluators", re.IGNORECASE),
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).casefold().strip()


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize text and keep a map from normalized character -> original index."""
    chars: list[str] = []
    offsets: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if previous_space:
                continue
            chars.append(" ")
            offsets.append(index)
            previous_space = True
            continue
        for folded in char.casefold():
            chars.append(folded)
            offsets.append(index)
        previous_space = False
    return "".join(chars), offsets


def _span_evidence(text: str, start: int, end: int) -> dict:
    return {"quote": text[start:end], "start": start, "end": end}


def _find_evidence(text: str, value: str) -> dict | None:
    """Locate ``value`` in ``text``; return quote + offsets when it is supported.

    Accepts a normalized substring match, or (for values longer than 3
    characters) at least 80% token overlap.
    """
    normalized_value = _normalize(value)
    if not normalized_value:
        return None

    normalized_text, offsets = _normalize_with_offsets(text)
    position = normalized_text.find(normalized_value)
    if position >= 0:
        start = offsets[position]
        end = offsets[position + len(normalized_value) - 1] + 1
        return _span_evidence(text, start, end)

    if len(normalized_value) <= 3:
        return None

    tokens = list(dict.fromkeys(_TOKEN_RE.findall(normalized_value)))
    if not tokens:
        return None

    found: list[tuple[int, int]] = []
    for token in tokens:
        match = re.search(re.escape(token), normalized_text)
        if match:
            found.append((match.start(), match.end()))

    if len(found) / len(tokens) < 0.8:
        return None

    start = offsets[min(item[0] for item in found)]
    end = offsets[max(item[1] for item in found) - 1] + 1
    return _span_evidence(text, start, end)


def _combined_text(candidate: dict) -> str:
    title = candidate.get("title") or candidate.get("name") or candidate.get("full_name") or ""
    body = " ".join(
        str(candidate.get(k, "")) for k in ("summary", "abstract", "description", "readme", "text")
    )
    parts = [str(title), body]
    for key in (
        "license", "venue", "publication_venue", "publicationVenue", "journal", "conference",
        "published", "year", "repo_url", "code_url", "dataset_url",
    ):
        value = candidate.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    return "\n".join(part for part in parts if part)


def _explicit_venue(candidate: dict) -> str:
    for key in VENUE_FIELD_KEYS:
        value = candidate.get(key)
        if value:
            return str(value)
    return ""


def _mark_unverified(meta: ExtractedMetadata, key: str) -> None:
    if key not in meta.unverified:
        meta.unverified.append(key)


def _mark_derived(meta: ExtractedMetadata, key: str) -> None:
    if key not in meta.derived:
        meta.derived.append(key)


def extract_rules(candidate: dict) -> ExtractedMetadata:
    title = candidate.get("title") or candidate.get("name") or candidate.get("full_name") or ""
    combined = _combined_text(candidate)

    meta = ExtractedMetadata(name=str(title).strip())
    url = candidate.get("url") or candidate.get("pdf_url") or candidate.get("html_url") or ""
    meta.source_url = url

    meta.code_url = str(candidate.get("repo_url") or candidate.get("code_url") or "")
    for match in _URL_RE.finditer(combined):
        found = match.group(0)
        if not meta.code_url and any(host in found for host in ("github.com", "gitlab.com", "bitbucket.org")):
            meta.code_url = found
        if not meta.data_url and any(host in found for host in ("huggingface.co/datasets", "zenodo.org", "doi.org/10.5281")):
            meta.data_url = found
    if not meta.data_url and candidate.get("dataset_url"):
        meta.data_url = str(candidate["dataset_url"])

    license_match = _LICENSE_RE.search(combined)
    if license_match:
        meta.license = license_match.group(0)
        meta.evidence["license"] = _span_evidence(combined, license_match.start(), license_match.end())
    elif candidate.get("license"):
        value = str(candidate["license"])
        evidence = _find_evidence(combined, value)
        if evidence:
            meta.license = value
            meta.evidence["license"] = evidence
        else:
            _mark_unverified(meta, "license")
    else:
        _mark_unverified(meta, "license")

    metric_match = _METRIC_RE.search(combined)
    if metric_match:
        meta.metric = metric_match.group(0)
        meta.evidence["metric"] = _span_evidence(combined, metric_match.start(), metric_match.end())
    else:
        _mark_unverified(meta, "metric")

    for pattern in _JUDGE_PATTERNS:
        judge_match = pattern.search(combined)
        if judge_match:
            if judge_match.group(0).casefold().startswith("human"):
                meta.judge = "human evaluators"
            else:
                meta.judge = "LLM judge"
            meta.evidence["judge"] = _span_evidence(combined, judge_match.start(), judge_match.end())
            break

    size_match = _SIZE_RE.search(combined)
    if size_match:
        meta.dataset_size = size_match.group(0)
        meta.evidence["dataset_size"] = _span_evidence(combined, size_match.start(), size_match.end())

    explicit_venue = _explicit_venue(candidate)
    if explicit_venue:
        evidence = _find_evidence(combined, explicit_venue)
        if evidence:
            meta.evidence["venue"] = evidence
        lowered_venue = explicit_venue.casefold()
        if PREPRINT_RE.search(lowered_venue):
            meta.peer_reviewed = False
        elif any(venue in lowered_venue for venue in VENUES):
            meta.peer_reviewed = True

    published = candidate.get("published") or candidate.get("year") or candidate.get("published_at")
    if published:
        meta.date = str(published)
        evidence = _find_evidence(combined, meta.date)
        if evidence:
            meta.evidence["date"] = evidence

    task_match = re.search(r"\bbenchmark\b|\bwe evaluate\b|\bwe propose an evaluation\b", combined, re.IGNORECASE)
    if task_match:
        meta.task_format = "evaluation benchmark"
        _mark_derived(meta, "task_format")
        meta.evidence["task_format"] = _span_evidence(combined, task_match.start(), task_match.end())
    measure_match = re.search(r"\bcreativ\w*", combined, re.IGNORECASE)
    if measure_match:
        meta.measures = "creativity-related capability"
        _mark_derived(meta, "measures")
        meta.evidence["measures"] = _span_evidence(combined, measure_match.start(), measure_match.end())

    if meta.peer_reviewed is None:
        _mark_unverified(meta, "peer_reviewed")

    return meta


_EXTRACT_PROMPT = """Extract benchmark metadata from the source below. Respond with ONLY a JSON object:
{{"name": str, "measures": str, "task_format": str, "metric": str, "judge": str,
  "dataset_size": str, "license": str, "code_url": str, "data_url": str, "date": str,
  "peer_reviewed": bool | null}}
Only include facts stated in the source. Use null/empty string when not stated.

SOURCE URL: {url}
SOURCE TEXT:
{text}
"""

_STRING_FIELDS = (
    "name", "measures", "task_format", "metric", "judge", "dataset_size",
    "license", "code_url", "data_url", "date",
)


async def extract_with_llm(candidate: dict, provider, model: str) -> ExtractedMetadata:
    """LLM-assisted extraction, rule-checked: unsupported values are marked unverified."""
    base = extract_rules(candidate)
    text = _combined_text(candidate)
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
        _mark_unverified(base, "llm_extraction:unparseable")
        return base

    for key in _STRING_FIELDS:
        value = data.get(key)
        if value in (None, "", "null") or not isinstance(value, str):
            continue
        evidence = _find_evidence(text, value)
        if evidence is None:
            _mark_unverified(base, key)
            continue
        if hasattr(base, key) and not getattr(base, key):
            setattr(base, key, value)
            base.evidence[key] = evidence
            if key in base.unverified:
                base.unverified.remove(key)
            if key in base.derived:
                base.derived.remove(key)

    value = data.get("peer_reviewed")
    if isinstance(value, bool):
        base.peer_reviewed = value
        if "peer_reviewed" in base.unverified:
            base.unverified.remove("peer_reviewed")

    return base


def extract(candidate: dict) -> ExtractedMetadata:
    """Synchronous rule-based extraction (no API calls)."""
    return extract_rules(candidate)
