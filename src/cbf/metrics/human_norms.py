"""Strict loader and z-based percentile for sourced human-norm data.

A norm file lives at ``data/human_norms/<instrument>.json`` and must contain:

    instrument, source_url, license, retrieval_date, n, mean, sd
    higher_is_better (optional, defaults to true)

Missing or malformed data raises :class:`HumanNormUnavailable`; no baseline is
ever invented. No dataset ships with this repository.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

DEFAULT_NORMS_DIR = Path("data/human_norms")
REQUIRED_KEYS = ("instrument", "source_url", "license", "retrieval_date", "n", "mean", "sd")


class HumanNormUnavailable(RuntimeError):
    """Raised when a human-norm baseline is missing or malformed."""


def _validate(data: Any, instrument: str, path: Path) -> dict:
    if not isinstance(data, dict):
        raise HumanNormUnavailable(f"Human norm file {path} must contain a JSON object")

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise HumanNormUnavailable(
            f"Human norm file {path} is missing required key(s): {', '.join(missing)}"
        )

    for key in ("instrument", "source_url", "license", "retrieval_date"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise HumanNormUnavailable(
                f"Human norm file {path} has an empty or non-string {key!r}"
            )

    if data["instrument"] != instrument:
        raise HumanNormUnavailable(
            f"Human norm file {path} declares instrument {data['instrument']!r}, expected {instrument!r}"
        )

    try:
        datetime.fromisoformat(str(data["retrieval_date"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HumanNormUnavailable(
            f"Human norm file {path} has an invalid retrieval_date {data['retrieval_date']!r}"
        ) from exc

    n = data["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise HumanNormUnavailable(f"Human norm file {path} has invalid n={n!r}; expected a positive integer")

    for key in ("mean", "sd"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HumanNormUnavailable(f"Human norm file {path} has invalid {key}={value!r}")
    if data["sd"] <= 0:
        raise HumanNormUnavailable(f"Human norm file {path} has non-positive sd={data['sd']!r}")

    if "higher_is_better" in data and not isinstance(data["higher_is_better"], bool):
        raise HumanNormUnavailable(
            f"Human norm file {path} has non-boolean higher_is_better={data['higher_is_better']!r}"
        )

    return data


def load_norm(instrument: str, norms_dir: str | Path = DEFAULT_NORMS_DIR) -> dict:
    """Load and validate one instrument norm; raise HumanNormUnavailable on any problem."""
    if not instrument or Path(instrument).name != instrument:
        raise HumanNormUnavailable(f"Invalid instrument name {instrument!r}")

    directory = Path(norms_dir)
    path = directory / f"{instrument}.json"
    if not path.exists():
        raise HumanNormUnavailable(
            f"No human-norm baseline for {instrument!r} at {path}; "
            "provide a sourced file (instrument, source_url, license, retrieval_date, n, mean, sd)"
        )

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanNormUnavailable(f"Could not read human-norm file {path}: {exc}") from exc

    return _validate(data, instrument, path)


def percentile(
    score: float,
    instrument: str,
    norms_dir: str | Path = DEFAULT_NORMS_DIR,
    *,
    higher_is_better: bool | None = None,
) -> float:
    """Return the z-based percentile (0..100) for ``score`` against a sourced norm.

    ``higher_is_better`` overrides the file's optional setting. A missing
    baseline always raises :class:`HumanNormUnavailable`; no fallback is used.
    """
    norm = load_norm(instrument, norms_dir)
    if higher_is_better is None:
        higher_is_better = bool(norm.get("higher_is_better", True))

    mean = float(norm["mean"])
    sd = float(norm["sd"])
    z = (float(score) - mean) / sd
    if not higher_is_better:
        z = -z

    value = NormalDist().cdf(z) * 100.0
    return max(0.0, min(100.0, value))
