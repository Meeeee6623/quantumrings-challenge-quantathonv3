"""Plot the four-model comparison at dot 1 of the feature progression.

Run: uv run --locked --extra report python research/plot_basic_counts_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "research/presentation_figures"
REPORT = json.loads((FIGURES / "model_sweep.json").read_text())
PACK = REPORT["packs"]["basic"]
MODELS = [
    ("ridge", "Ridge regression"),
    ("hist_boost", "Histogram gradient boosting"),
    ("random_forest", "Random forest"),
    ("extra_trees", "ExtraTrees"),
]


def main() -> None:
    scores = np.array([100 * PACK["models"][key]["score"] for key, _ in MODELS])
    assert len(PACK["columns"]) == 29
    assert int(np.argmax(scores)) == 3

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=.27, right=.93, top=.78, bottom=.18)

    y = np.arange(4)
    colors = [plt.get_cmap("tab10")(i) for i in (0, 1, 4, 2)]
    bars = ax.barh(y, scores, height=.59, color=colors, edgecolor="white", linewidth=1.2)
    bars[3].set_edgecolor("#176A2B")
    bars[3].set_linewidth(2.1)
    for i, value in enumerate(scores):
        ax.text(value + 1.25, i, f"{value:.2f}%", va="center", ha="left",
                fontsize=19, fontweight="bold" if i == 3 else "normal",
                color="#176A2B" if i == 3 else "#23384D")

    ax.set_yticks(y, [label for _, label in MODELS], fontsize=17)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis="x", labelsize=12, colors="#607183", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.grid(axis="x", color="#DFE7ED", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Official duration score (%)", fontsize=14, color="#23384D", labelpad=13)

    fig.text(.07, .955, "Dot 1 · Basic QASM counts", fontsize=31,
             fontweight="bold", color="#17324D", ha="left", va="top")
    fig.text(.07, .885,
             "Same 29 inputs and five circuit-grouped folds for every model · ExtraTrees wins",
             fontsize=16, color="#607183", ha="left", va="top")
    fig.text(.07, .08,
             "Inputs: qubits, operation counts, depth, measurements, resets, file size, log copies, and simulator-setting indicators.",
             fontsize=12, color="#607183", ha="left", va="bottom")
    fig.text(.07, .045,
             "Source: model_sweep.json · 1,497 labeled runs · Predicted log10(seconds); bars show grouped out-of-fold score.",
             fontsize=11, color="#607183", ha="left", va="bottom")

    stem = FIGURES / "09_basic_counts_model_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')} and {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
