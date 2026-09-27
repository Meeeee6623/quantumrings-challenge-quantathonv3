"""Reproduce the split sizes and final OOF subgroup numbers in the handoff.

Run: uv run --locked python research/summarize_validation_context.py
Reads only checked-in CSV/JSON evidence; does not fit or modify a model.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def csv_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]], key, split: str) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        actual = float(row["actual_s"])
        predicted = float(row[f"{split}_pred_s"])
        if row["status"] == "timeout":
            predicted = min(predicted, 14400.0)
        error_factor = max(predicted / actual, actual / predicted)
        score = max(0.0, 1.0 - abs(math.log10(predicted / actual)) / 2.0)
        grouped[key(row)].append((row["filename"], score, error_factor, row["status"]))
    return {
        str(name): {
            "circuits": len({item[0] for item in values}),
            "labeled_rows": len(values),
            "timeouts": sum(item[3] == "timeout" for item in values),
            "score": sum(item[1] for item in values) / len(values),
            "over_10x": sum(item[2] > 10 for item in values),
        }
        for name, values in sorted(grouped.items(), key=lambda pair: str(pair[0]))
    }


def size_bin(features: dict, filename: str) -> str:
    size = features[filename]["qasm_bytes"]
    if size < 1_000_000:
        return "under_1m"
    if size < 10_000_000:
        return "1m_to_10m"
    if size < 40_000_000:
        return "10m_to_40m"
    return "over_40m"


def main() -> None:
    assignments = {row["filename"]: row for row in csv_rows("research/comparison_folds.csv")}
    labels = csv_rows("runtime-data.csv")
    predictions = csv_rows("research/production_model_oof.csv")
    features = json.loads((ROOT / "research/features.json").read_text())
    assert len(assignments) == len(features) == 532
    assert len(labels) == len(predictions) == 1497
    assert [(row["filename"], row["threshold"]) for row in labels] == [
        (row["filename"], row["threshold"]) for row in predictions
    ]

    result = {
        "circuits": len(assignments),
        "labeled_rows": len(labels),
        "explicit_timeouts": sum(row["status"] == "timeout" for row in labels),
        "distinct_group_signatures": len({row["feature_signature_sha1"] for row in assignments.values()}),
        "matched_folds": summarize(
            predictions, lambda row: assignments[row["filename"]]["matched_fold"], "matched"
        ),
        "structural_folds": summarize(
            predictions, lambda row: assignments[row["filename"]]["structural_stress_fold"], "structural"
        ),
        "matched_by_setting": summarize(predictions, lambda row: row["threshold"], "matched"),
        "matched_by_status": summarize(predictions, lambda row: row["status"], "matched"),
        "matched_by_size": summarize(
            predictions, lambda row: size_bin(features, row["filename"]), "matched"
        ),
        "structural_by_size": summarize(
            predictions, lambda row: size_bin(features, row["filename"]), "structural"
        ),
        "matched_overall": summarize(predictions, lambda row: "all", "matched")["all"],
        "structural_overall": summarize(predictions, lambda row: "all", "structural")["all"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
