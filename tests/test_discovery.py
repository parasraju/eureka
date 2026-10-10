import sys

sys.path.insert(0, r'D:\p\eureka\src')

import pytest

from cbf.discovery.extractor import ExtractedMetadata, extract
from cbf.discovery.ranker import rank, rank_candidate
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


def test_extract_rules_fields():
    meta = extract(make_candidate())
    assert isinstance(meta, ExtractedMetadata)
    assert meta.license == "MIT"
    assert meta.source_url.startswith("http")
    assert meta.metric
    assert "preprint" not in meta.evidence or meta.peer_reviewed is not None


def test_registry_workflow(tmp_path):
    db = RegistryDB(str(tmp_path / "reg.db"))
    try:
        r1 = rank_candidate(make_candidate(), min_score=0.3)
        r2 = rank_candidate(make_candidate(id="test-2", title="Divergent Thinking Across Models",
                                           summary="Benchmark measuring divergent association, originality and creativity in language models."),
                            min_score=0.3)
        count = db.upsert([r1, r2])
        assert count == 2

        got = db.get("test-1")
        assert got is not None
        assert got.status == r1.status
        assert got.adapter_path == ""

        assert [e.id for e in db.search()] == sorted(["test-1", "test-2"])
        assert [e.id for e in db.search(status="candidate")] == sorted(["test-1", "test-2"])
        assert db.approved() == []

        db.set_status("test-1", "approved", reason="reviewed")
        assert db.get("test-1").status == "approved"
        assert [e.id for e in db.approved()] == ["test-1"]

        with pytest.raises(ValueError):
            db.set_status("test-1", "bogus")
        with pytest.raises(KeyError):
            db.set_status("nope", "approved")

        db.set_adapter("test-2", "src/cbf/benchmarks/adapters/test_2.py")
        assert db.get("test-2").status == "adapter_drafted"
        assert db.get("test-2").adapter_path.endswith("test_2.py")

        # upsert preserves status/approval
        db.upsert([r1])
        assert db.get("test-1").status == "approved"

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
