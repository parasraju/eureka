import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from cbf.report.charts import radar_chart
from cbf.report.compare import (
    DIMENSION_BOUNDS,
    creativity_profile,
    render_comparison_markdown,
    render_profiles_markdown,
)
from cbf.report.generator import (
    compute_scores_from_run,
    generate_json_report,
    generate_leaderboard,
    generate_leaderboard_markdown,
    generate_markdown_report,
    load_run_results,
)


def _result(benchmark, condition, metrics, confidence_intervals=None, metadata=None):
    return {
        "benchmark": benchmark,
        "condition": condition,
        "model": "test-model",
        "raw": {
            "benchmark_id": benchmark,
            "model": "test-model",
            "items": [],
            "call_ids": [],
            "metadata": {"condition": condition},
        },
        "scores": {
            "benchmark_id": benchmark,
            "model": "test-model",
            "metrics": metrics,
            "confidence_intervals": confidence_intervals or {},
            "metadata": metadata or {"condition": condition},
        },
    }


def _write_run(
    tmp_path,
    results=None,
    run_id="run1",
    suite="dat",
    status="completed",
    write_results=True,
    total_calls=12,
    total_cost=0.5,
):
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "started_utc": "2026-01-01T00:00:00+00:00",
        "ended_utc": "2026-01-01T00:10:00+00:00",
        "models": ["test-model"],
        "provider_versions": {},
        "config_hash": "",
        "prompt_set_hashes": {},
        "embedding_model": "",
        "judge_models": [],
        "conditions": {},
        "total_cost_usd": total_cost,
        "total_calls": total_calls,
        "status": status,
        "suite": suite,
        "error": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    if write_results:
        with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for entry in results or []:
                f.write(json.dumps(entry) + "\n")

    return run_id


def test_radar_chart_creates_file(tmp_path):
    out = str(tmp_path / "radar.png")
    models = {
        "gpt-4o": {"divergent_thinking": 60.0, "diversity": 40.0},
        "claude": {"divergent_thinking": 70.0, "diversity": 55.0},
    }
    radar_chart(models, out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_radar_chart_mismatched_dims(tmp_path):
    models = {
        "a": {"x": 1.0, "y": 2.0},
        "b": {"x": 1.0},
    }
    with pytest.raises(ValueError):
        radar_chart(models, str(tmp_path / "x.png"))


def test_radar_chart_empty():
    with pytest.raises(ValueError):
        radar_chart({}, "x.png")


def test_render_profiles_markdown():
    profiles = [
        {"model": "a", "dimensions": {"diversity": 42.5, "chain_validity": 80.0}, "composite": 61.2, "cost_usd": 0.5, "error": None},
        {"model": "b", "dimensions": {"diversity": 33.3, "chain_validity": 66.0}, "composite": 49.6, "cost_usd": 0.7, "error": None},
    ]
    md = render_profiles_markdown(profiles)
    assert "| Model |" in md
    assert "| a |" in md
    assert "| Composite |" in md


def test_render_comparison_markdown():
    cmp = {
        "compared": 2,
        "rows": {
            "dat": [
                {
                    "run_id": "r1", "model": "m1",
                    "metrics": {"dat_score": 55.0},
                    "confidence_intervals": {"dat_score": (50.0, 60.0)},
                    "cost_usd": 0.1, "calls": 50, "started_utc": "x",
                },
                {
                    "run_id": "r2", "model": "m2",
                    "metrics": {"dat_score": 60.0},
                    "confidence_intervals": {"dat_score": (55.0, 65.0)},
                    "cost_usd": 0.2, "calls": 50, "started_utc": "x",
                },
            ]
        },
    }
    md = render_comparison_markdown(cmp)
    assert "## dat" in md
    assert "r1" in md
    assert "r2" in md


def test_profile_missing_run(tmp_path):
    p = creativity_profile("does-not-exist", str(tmp_path))
    assert "error" in p


def test_bounds_are_documented():
    assert set(DIMENSION_BOUNDS) == {
        "divergent_thinking", "fluency", "flexibility", "originality",
        "diversity", "chain_validity", "repurposing", "writing_elo",
    }
    for lo, hi in DIMENSION_BOUNDS.values():
        assert lo < hi


def test_load_run_results_missing_results_file(tmp_path):
    run_id = _write_run(tmp_path, write_results=False)
    data = load_run_results(run_id, str(tmp_path))
    assert data["error"] == (
        f"Run {run_id} has no persisted results (results.jsonl). "
        "Re-run with the current version."
    )


def test_load_run_results_ignores_manifest_suite(tmp_path):
    run_id = _write_run(
        tmp_path,
        results=[_result("aut", "neutral", {"fluency": 3.0})],
        suite="dat",
    )
    data = load_run_results(run_id, str(tmp_path))
    assert data["results"][0]["benchmark"] == "aut"

    md = generate_markdown_report(run_id, str(tmp_path))
    assert "## aut / neutral" in md
    assert "## dat /" not in md


def test_compute_scores_from_persisted_results(tmp_path):
    run_id = _write_run(
        tmp_path,
        results=[
            _result("dat", "neutral", {"dat_score": 55.0}, {"dat_score": (50.0, 60.0)}),
            _result("aut", "creative", {"fluency": 7.0}, {"fluency": (6.0, 8.0)}),
        ],
    )
    scores = compute_scores_from_run(load_run_results(run_id, str(tmp_path)))

    assert scores["status"] == "completed"
    assert scores["model"] == "test-model"
    assert len(scores["results"]) == 2
    assert scores["results"][0]["benchmark"] == "dat"
    assert scores["results"][0]["confidence_intervals"]["dat_score"] == [50.0, 60.0]
    assert scores["results"][1]["condition"] == "creative"
    assert scores["run_info"]["total_calls"] == 12
    # Legacy single-benchmark keys remain for report/compare.py.
    assert scores["benchmark"] == "dat"
    assert scores["metrics"] == {"dat_score": 55.0}


def test_markdown_report_renders_every_benchmark_and_condition(tmp_path):
    run_id = _write_run(
        tmp_path,
        results=[
            _result("dat", "neutral", {"dat_score": 55.0}, {"dat_score": (50.0, 60.0)}),
            _result("dat", "creative", {"dat_score": 65.0}, {"dat_score": (60.0, 70.0)}),
            _result("aut", "neutral", {"fluency": 4.0}, {"fluency": (3.0, 5.0)}),
        ],
    )
    md = generate_markdown_report(run_id, str(tmp_path))

    assert "**Run ID:** run1" in md
    assert "**Model(s):** test-model" in md
    assert "**Status:** completed" in md
    assert "**Total Cost:** $0.5000" in md
    assert "**Total API Calls:** 12" in md
    assert "## dat / neutral" in md
    assert "## dat / creative" in md
    assert "## aut / neutral" in md
    assert "95% CI: 50.000 - 60.000" in md


def test_markdown_report_reports_missing_results(tmp_path):
    run_id = _write_run(tmp_path, write_results=False)
    md = generate_markdown_report(run_id, str(tmp_path))
    assert "no persisted results (results.jsonl)" in md


def test_json_report_schema_is_json_safe(tmp_path):
    run_id = _write_run(
        tmp_path,
        results=[
            _result("dat", "neutral", {"dat_score": 55.0}, {"dat_score": (50.0, 60.0)}),
            _result("aut", "creative", {"fluency": 7.0}, {"fluency": (6.0, 8.0)}),
        ],
    )
    report = generate_json_report(run_id, str(tmp_path))

    assert set(report) == {
        "run_id", "model", "status", "started_utc", "ended_utc",
        "total_cost_usd", "total_calls", "results", "generated_at",
    }
    assert report["run_id"] == "run1"
    assert report["model"] == "test-model"
    assert report["status"] == "completed"
    assert report["total_calls"] == 12
    assert len(report["results"]) == 2
    assert set(report["results"][0]) == {
        "benchmark", "condition", "metrics", "confidence_intervals", "metadata",
    }
    assert isinstance(report["results"][0]["confidence_intervals"]["dat_score"], list)
    json.dumps(report)


def test_leaderboard_means_across_conditions(tmp_path):
    run_id = _write_run(
        tmp_path,
        results=[
            _result("dat", "neutral", {"dat_score": 60.0, "valid_samples": 10}),
            _result("dat", "creative", {"dat_score": 80.0, "valid_samples": 20}),
        ],
    )
    board = generate_leaderboard([run_id], str(tmp_path))

    entry = board["test-model"]["dat"]
    assert entry["metrics"]["dat_score"] == pytest.approx(70.0)
    assert entry["metrics"]["valid_samples"] == pytest.approx(15.0)
    assert entry["conditions"] == ["creative", "neutral"]

    md = generate_leaderboard_markdown(board)
    assert "`dat_score`" in md
    assert "| Conditions |" in md
    assert "| 2 |" in md

def test_collect_texts_by_model_dedupes_and_skips_errors(monkeypatch):
    from cbf.report import generator as gen

    runs = {
        "r1": {"results": [
            {"benchmark": "writing", "model": "m1", "raw": {"items": [
                {"text": "alpha"}, {"error": "boom"}, {"text": "beta"},
            ]}},
        ]},
        "r2": {"results": [
            {"benchmark": "writing", "model": "m1", "raw": {"items": [{"text": "alpha"}]}},
            {"benchmark": "dat", "model": "m2", "raw": {"items": [{"text": "gamma"}]}},
        ]},
        "bad": {"error": "Run bad not found"},
    }
    monkeypatch.setattr(gen, "load_run_results", lambda run_id, runs_dir="runs": runs[run_id])

    texts, errors = gen.collect_texts_by_model(["r1", "r2", "bad"], benchmark="writing")
    assert texts == {"m1": ["alpha", "beta"]}
    assert errors == ["bad: Run bad not found"]

    all_texts, _ = gen.collect_texts_by_model(["r1", "r2"])
    assert all_texts["m2"] == ["gamma"]
