import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, r'D:\p\eureka\src')

import pytest

from cbf.discovery.extractor import ExtractedMetadata, extract, extract_with_llm
from cbf.discovery.ranker import credibility_score, rank, rank_candidate, runnability_score
from cbf.discovery.registry_db import RegistryDB


def make_candidate(**overrides):
    c = {
        "id": "test-1",
        "source": "arxiv",
        "title": "Creativity Benchmark for Large Language Models",
        "summary": "We propose a benchmark evaluating creative writing divergence for LLMs using an LLM-judge rubric.",
        "published": "2026-09-01",
        "citationCount": 5,
        "url": "https://arxiv.org/abs/2609.00001",
        "license": "MIT",
        "repo_url": "https://github.com/example/repo",
    }
    c.update(overrides)
    return c


def test_rank_candidate_components():
    r = rank_candidate(make_candidate(), min_score=0.3)
    assert r.status == "candidate"
    assert r.relevance > 0.3
    assert r.runnability > 0.4
    assert r.score > 0.4
    assert 0 <= r.score <= 1


def test_rank_candidate_below_min_rejected():
    c = make_candidate(title="Some unrelated paper on aquatic mollusks",
                       summary="A study of gastropod locomotion in the intertidal zone.",
                       published="2026-09-01")
    r = rank_candidate(c, min_score=0.3)
    assert r.status == "rejected"


def test_rank_candidate_preprint():
    r = rank_candidate(make_candidate(summary="Preprint under review at arXiv."), min_score=0.3)
    assert "preprint" in " ".join(r.reasons).lower()


def test_rank_sorts_desc():
    low = rank_candidate(make_candidate(title="Boring unrelated title"))
    high = rank_candidate(make_candidate())
    ranked = rank([low, high], min_score=0.3)
    assert ranked[0].id == high.id


def test_runnability_only_counts_real_repositories():
    arxiv_html = {
        "id": "arxiv-only",
        "title": "Creativity Benchmark",
        "html_url": "https://arxiv.org/abs/2609.00002",
    }
    assert runnability_score(arxiv_html) == 0.0

    github_html = {"html_url": "https://github.com/example/repo"}
    assert runnability_score(github_html) == pytest.approx(0.40)

    explicit_repo = {"repo_url": "https://example.com/not-a-known-host"}
    assert runnability_score(explicit_repo) == pytest.approx(0.40)


def test_credibility_requires_explicit_venue_not_arxiv_mentions():
    text_only = {
        "title": "A creativity benchmark",
        "summary": "Posted on arXiv and under review.",
        "citationCount": 0,
    }
    assert credibility_score(text_only, peer_reviewed=None) == pytest.approx(0.20)

    explicit = {"venue": "NeurIPS 2026", "citationCount": 0}
    assert credibility_score(explicit, peer_reviewed=None) == pytest.approx(0.35)


def test_extract_rules_fields():
    meta = extract(make_candidate())
    assert isinstance(meta, ExtractedMetadata)
    assert meta.license == "MIT"
    assert meta.source_url.startswith("http")
    assert meta.metric
    assert meta.evidence["license"]["quote"] == "MIT"
    assert isinstance(meta.evidence["license"]["start"], int)
    assert isinstance(meta.evidence["license"]["end"], int)
    assert meta.evidence["license"]["end"] > meta.evidence["license"]["start"]
    assert "task_format" in meta.derived
    assert "measures" in meta.derived
    assert "preprint" not in meta.evidence or meta.peer_reviewed is not None


def test_extract_rules_do_not_use_arxiv_html_url_as_code():
    candidate = make_candidate(repo_url="", html_url="https://arxiv.org/abs/2609.00001")
    candidate["url"] = "https://arxiv.org/abs/2609.00001"
    meta = extract(candidate)
    assert meta.code_url == ""


async def test_extract_with_llm_rejects_unsupported_fields():
    candidate = {
        "id": "llm-test",
        "source": "arxiv",
        "title": "",
        "summary": "We propose a benchmark evaluating creative writing with an LLM judge.",
        "url": "https://arxiv.org/abs/2609.00003",
    }
    payload = (
        '{"name": "Fabricated Benchmark Name", "measures": "quantum entanglement theory", '
        '"task_format": "evaluation benchmark", "metric": "BLEU", '
        '"judge": "human evaluators", "license": "Apache-2.0", "date": "1999-01-01"}'
    )

    class StubProvider:
        async def complete(self, request):
            return SimpleNamespace(text=payload)

    meta = await extract_with_llm(candidate, StubProvider(), "fake:model")

    assert meta.name == ""
    assert meta.date == ""
    assert meta.license == ""
    assert meta.measures == "creativity-related capability"
    assert "name" in meta.unverified
    assert "date" in meta.unverified
    assert "license" in meta.unverified
    assert "measures" in meta.derived
    assert "task_format" in meta.derived


def test_registry_workflow(tmp_path):
    db = RegistryDB(str(tmp_path / "reg.db"))
    try:
        r1 = rank_candidate(make_candidate(), min_score=0.3)
        r2 = rank_candidate(make_candidate(id="test-2", title="Divergent Thinking Across Models",
                                           summary="Benchmark measuring divergent association, originality and creativity in language models."),
                            min_score=0.3)
        r3 = {
            "id": "test-3",
            "source": "arxiv",
            "title": "Seed Benchmark",
            "url": "https://arxiv.org/abs/2609.00003",
            "summary": "",
            "status": "seed",
        }
        count = db.upsert([r1, r2, r3])
        assert count == 3

        got = db.get("test-1")
        assert got is not None
        assert got.status == r1.status
        assert got.adapter_path == ""

        assert [e.id for e in db.search()] == sorted(["test-1", "test-2", "test-3"])
        assert [e.id for e in db.search(status="candidate")] == sorted(["test-1", "test-2"])
        assert db.approved() == []

        with pytest.raises(ValueError):
            db.set_status("test-1", "approved", reason="skipped steps")
        with pytest.raises(ValueError):
            db.set_status("test-3", "approved", reason="skipped steps")

        db.set_adapter("test-1", "src/cbf/benchmarks/adapters/test_1.py")
        assert db.get("test-1").status == "adapter_drafted"
        db.set_status("test-1", "approved", reason="reviewed")
        assert db.get("test-1").status == "approved"
        assert [e.id for e in db.approved()] == ["test-1"]

        with pytest.raises(ValueError):
            db.set_status("test-1", "candidate")

        first_seen = db.get("test-1").first_seen
        db.upsert([r1])
        assert db.get("test-1").status == "approved"
        assert db.get("test-1").first_seen == first_seen

        db.set_adapter("test-2", "src/cbf/benchmarks/adapters/test_2.py")
        db.set_status("test-2", "approved", reason="reviewed")
        with pytest.raises(ValueError):
            db.set_status("test-2", "rejected")
        db.set_status("test-2", "rejected", reason="not a fit")
        assert db.get("test-2").status == "rejected"
        with pytest.raises(ValueError):
            db.set_status("test-2", "approved")
        assert db.get("test-2").adapter_path.endswith("test_2.py")

        with pytest.raises(ValueError):
            db.set_status("test-1", "bogus")
        with pytest.raises(KeyError):
            db.set_status("nope", "approved")

        digest = db.digest(days=7)
        assert digest.startswith("# Benchmark discovery digest")
        assert "Creativity Benchmark for Large Language Models" in digest
    finally:
        db.close()


def test_registry_digest_empty(tmp_path):
    db = RegistryDB(str(tmp_path / "reg2.db"))
    try:
        assert "No new benchmarks" in db.digest(days=7)
    finally:
        db.close()


def test_registry_new_since_normalizes_timestamps(tmp_path):
    db = RegistryDB(str(tmp_path / "reg3.db"))
    try:
        recent = rank_candidate(make_candidate(id="recent"), min_score=0.3)
        old = rank_candidate(make_candidate(id="old"), min_score=0.3)
        db.upsert([recent, old])
        old_timestamp = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        recent_timestamp = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        db.conn.execute("UPDATE benchmarks SET first_seen = ? WHERE id = 'old'", (old_timestamp,))
        db.conn.execute("UPDATE benchmarks SET first_seen = ? WHERE id = 'recent'", (recent_timestamp,))
        db.conn.commit()

        assert [e.id for e in db.new_since(days=7)] == ["recent"]
    finally:
        db.close()
