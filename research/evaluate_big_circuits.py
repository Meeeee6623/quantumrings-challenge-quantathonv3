"""Evaluate size-dependent prediction errors using the saved out-of-fold runs.

Run from the repository root:
    uv run --locked --extra report python research/evaluate_big_circuits.py
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research"
CAP = 14400.0
VIEWS = {
    "baseline_matched": "baseline_matched_pred_s",
    "current_matched": "filtered_plus_angles_matched_pred_s",
    "current_structural": "filtered_plus_angles_structural_pred_s",
}
BINS = ("<1 MB", "1–10 MB", "10–40 MB", ">40 MB")


def size_bin(n_bytes):
    if n_bytes < 1_000_000:
        return BINS[0]
    if n_bytes < 10_000_000:
        return BINS[1]
    if n_bytes <= 40_000_000:
        return BINS[2]
    return BINS[3]


def row_error(row, column):
    actual = float(row["actual_s"])
    predicted = float(row[column])
    if row["status"] == "timeout":
        predicted = min(predicted, CAP)
    signed_log_error = math.log10(predicted / actual)
    return {
        "score": max(0.0, 1.0 - abs(signed_log_error) / 2.0),
        "factor": 10 ** abs(signed_log_error),
        "signed_log10_error": signed_log_error,
    }


def summarize(rows, parse_by_name):
    names = sorted({row["filename"] for row in rows})
    result = {
        "circuits": len(names),
        "labeled_rows": len(rows),
        "timeouts": sum(row["status"] == "timeout" for row in rows),
        "parse_s_median": float(np.median([parse_by_name[name] for name in names])),
        "parse_s_max": max(parse_by_name[name] for name in names),
    }
    for view, column in VIEWS.items():
        errors = [row_error(row, column) for row in rows]
        factors = np.array([error["factor"] for error in errors])
        result[view] = {
            "mean_score": float(np.mean([error["score"] for error in errors])),
            "median_factor_error": float(np.median(factors)),
            "p95_factor_error": float(np.quantile(factors, 0.95)),
            "over_3x_rows": int((factors > 3).sum()),
            "over_10x_rows": int((factors > 10).sum()),
            "over_100x_rows": int((factors > 100).sum()),
        }
    result["by_threshold"] = {}
    for threshold in (16, 64, 512):
        subset = [row for row in rows if int(row["threshold"]) == threshold]
        if subset:
            errors = [row_error(row, VIEWS["current_matched"]) for row in subset]
            result["by_threshold"][str(threshold)] = {
                "rows": len(subset),
                "mean_score": float(np.mean([error["score"] for error in errors])),
                "over_10x_rows": sum(error["factor"] > 10 for error in errors),
            }
    return result


def plot_size_scores(summaries):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    x = np.arange(len(BINS))
    width = 0.25
    for offset, view, label, color in (
        (-1, "baseline_matched", "231-feature baseline, matched", "#adb5bd"),
        (0, "current_matched", "Current, matched", "#1f77b4"),
        (1, "current_structural", "Current, structural stress", "#d97706"),
    ):
        ax.bar(x + offset * width,
               [summaries[bin_name][view]["mean_score"] for bin_name in BINS],
               width, color=color, label=label)
    ax.set_xticks(x, [f"{name}\n{summaries[name]['circuits']} circuits / "
                      f"{summaries[name]['labeled_rows']} labels" for name in BINS])
    ax.set_ylabel("Mean challenge score (higher is better)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Runtime accuracy by QASM size and validation split")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "big_circuit_score_by_size.png", dpi=170)
    plt.close(fig)


def plot_fastpath(rows):
    names = sorted({row["filename"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    shown_timeout = False
    for ax, name in zip(axes.flat, names):
        subset = sorted((row for row in rows if row["filename"] == name),
                        key=lambda row: int(row["threshold"]))
        x = np.arange(len(subset))
        actual = [float(row["actual_s"]) for row in subset]
        prediction = [float(row[VIEWS["current_matched"]]) for row in subset]
        ax.plot(x, actual, color="#222222", marker="o", linewidth=1.8,
                label="Measured / timeout target")
        ax.plot(x, prediction, color="#1f77b4", marker="s", linewidth=1.8,
                label="Matched-fold prediction")
        for pos, row in zip(x, subset):
            if row["status"] == "timeout":
                ax.scatter([pos], [CAP], color="#d97706", marker="^", s=75,
                           zorder=5, label="Timeout (censored)" if not shown_timeout else None)
                shown_timeout = True
        ax.set_title(name)
        ax.set_xticks(x, [row["threshold"] for row in subset])
        ax.set_yscale("log")
        ax.set_ylim(100, 25000)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("Threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Seconds (log scale)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0))
    fig.suptitle("All four >40 MB circuits: actual versus held-out prediction")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(HERE / "big_circuit_fastpath_predictions.png", dpi=170)
    plt.close(fig)


def main():
    features = json.loads((HERE / "features.json").read_text())
    walks = json.loads((HERE / "chi_walk_angle_cache.json").read_text())["tables"]
    with (HERE / "rotation_filter_oof.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    with (HERE / "training_submission_rotation_filter.csv").open(newline="") as file:
        parse_by_name = {row["filename"]: float(row["parse_s"])
                         for row in csv.DictReader(file)}
    assert len(rows) == 1497 and len(features) == len(parse_by_name) == 532
    by_bin = {bin_name: [] for bin_name in BINS}
    for row in rows:
        by_bin[size_bin(features[row["filename"]]["qasm_bytes"])].append(row)
    summaries = {bin_name: summarize(group, parse_by_name)
                 for bin_name, group in by_bin.items()}
    fast_rows = sorted(by_bin[BINS[3]],
                       key=lambda row: (row["filename"], int(row["threshold"])))
    assert len(fast_rows) == 12
    assert all(features[row["filename"]]["huge_fast_path"] == 1 and
               walks[row["filename"]] is None for row in fast_rows)
    fast_details = []
    for row in fast_rows:
        name = row["filename"]
        error = row_error(row, VIEWS["current_matched"])
        fast_details.append({
            "filename": name,
            "qasm_mb": features[name]["qasm_bytes"] / 1_000_000,
            "threshold": int(row["threshold"]),
            "status": row["status"],
            "actual_s": float(row["actual_s"]),
            "matched_pred_s": float(row[VIEWS["current_matched"]]),
            "structural_pred_s": float(row[VIEWS["current_structural"]]),
            "matched_factor_error": error["factor"],
            "matched_score": error["score"],
            "parse_s": parse_by_name[name],
        })
    extrapolated = [row for row in rows if walks[row["filename"]] is not None
                    and walks[row["filename"]].get("extrapolated", 0) > 0]
    result = {
        "source": "rotation_filter_oof.csv; training_submission_rotation_filter.csv for parser timing",
        "size_definition": "decoded QASM text length in Python characters (approximately bytes for ASCII QASM); fast path when >40,000,000 characters",
        "size_bins": summaries,
        "fast_path_rows": fast_details,
        "extrapolated_walk": summarize(extrapolated, parse_by_name),
    }
    (HERE / "big_circuit_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_size_scores(summaries)
    plot_fastpath(fast_rows)
    for bin_name, summary in summaries.items():
        print(bin_name, summary["circuits"], "circuits,",
              summary["labeled_rows"], "rows; matched score",
              round(summary["current_matched"]["mean_score"], 5),
              ">10x errors", summary["current_matched"]["over_10x_rows"],
              "stress score", round(summary["current_structural"]["mean_score"], 5))


if __name__ == "__main__":
    main()
