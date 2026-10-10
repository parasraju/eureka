"""Radar chart rendering for creativity profiles (matplotlib, headless Agg backend)."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


def radar_chart(
    models: dict[str, dict[str, float]],
    output_path: str,
    title: str = "Creativity Profile",
    filename_safe: bool = True,
) -> str:
    """Render a radar chart of 0-100 dimension scores per model.

    models: {model_id: {dimension: 0-100}}
    Returns the output file path. Raises ValueError if dimension sets do not match.
    """
    if not models:
        raise ValueError("No models to plot")
    dims = list(next(iter(models.values())).keys())
    if any(set(dims) != set(m.keys()) for m in models.values()):
        raise ValueError("Models must share the same dimension set for a radar plot")

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for model_id, scores in models.items():
        values = [float(scores[d]) for d in dims] + [float(scores[dims[0]])]
        ax.plot(angles, values, linewidth=2, label=model_id)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"])
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
