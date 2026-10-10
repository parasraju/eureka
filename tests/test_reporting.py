import sys

sys.path.insert(0, r'D:\p\eureka\src')

import os

import pytest

from cbf.report.charts import radar_chart
from cbf.report.compare import (
    DIMENSION_BOUNDS,
    creativity_profile,
    render_comparison_markdown,
    render_profiles_markdown,
)


def test_radar_chart_creates_file(tmp_path):
    out = str(tmp_path / "radar.png")
    models = {
        "gpt-4o": {"divergent_thinking": 60.0, "diversity": 40.0},
        "claude": {"divergent_thinking": 70.0, "diversity": 55.0},
    }
    radar_chart(models, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


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
