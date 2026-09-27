"""Probe a train-fold-only floor for reset-heavy, multi-control circuits.

This uses the saved full-union OOF predictions, so no regressor is refitted.
Each test circuit's reference is chosen only from the corresponding training
fold.  Run: uv run --locked python research/probe_reset_family_floor.py
"""

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research"
CAP = 14400.0
sys.path.insert(0,str(ROOT / 'quantathon-harness'))
from runtime_floors import eligible as trigger  # noqa: E402


def score(row, seconds):
    actual = float(row["actual_s"])
    if row["status"] == "timeout":
        seconds = min(seconds, CAP)
    return max(0.0, 1.0 - abs(math.log10(seconds / actual)) / 2.0)


def evaluate(rows, features, folds, split, alpha, cap_reference):
    fold_key = "matched_fold" if split == "matched" else "structural_stress_fold"
    pred_key = f"{split}_full_union_pred_s"
    current = np.array([float(row[pred_key]) for row in rows])
    updated = current.copy()
    changed = []
    reference_rows = [row for row in rows if trigger(features[row["filename"]])]
    for index, row in enumerate(rows):
        name = row["filename"]
        if not trigger(features[name]):
            continue
        operation_count = max(1.0, float(features[name]["ops"]))
        candidates = [reference for reference in reference_rows
                      if reference["threshold"] == row["threshold"]
                      and folds[reference["filename"]][fold_key] != folds[name][fold_key]]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda reference: abs(math.log(
            operation_count / max(1.0,float(features[reference["filename"]]["ops"])))))
        reference_target = float(nearest["actual_s"])
        if cap_reference:
            reference_target = min(CAP, reference_target)
        reference_ops = max(1.0, float(features[nearest["filename"]]["ops"]))
        floor = min(CAP, reference_target * (operation_count / reference_ops) ** alpha)
        updated[index] = max(current[index], floor)
        if updated[index] > current[index]:
            changed.append({
                "filename":name, "threshold":int(row["threshold"]),
                "status":row["status"], "actual_s":float(row["actual_s"]),
                "reference_circuit":nearest["filename"],
                "reference_ops":reference_ops, "reference_target_s":reference_target,
                "old_pred_s":float(current[index]), "floor_s":float(floor),
                "new_pred_s":float(updated[index]),
                "old_score":score(row,current[index]),
                "new_score":score(row,updated[index]),
            })
    old_score = np.array([score(row,seconds) for row,seconds in zip(rows,current)])
    new_score = np.array([score(row,seconds) for row,seconds in zip(rows,updated)])
    special = np.array([trigger(features[row["filename"]]) for row in rows])
    diff = new_score - old_score
    if split == "matched":
        units = np.array([row["filename"] for row in rows])
        bootstrap_unit = "circuit"
    else:
        units = np.array([folds[row["filename"]]["structural_cluster"] for row in rows])
        bootstrap_unit = "structural_cluster"
    unique, group = np.unique(units, return_inverse=True)
    sums = np.bincount(group, weights=diff, minlength=len(unique))
    counts = np.bincount(group, minlength=len(unique))
    rng = np.random.default_rng(41)
    draw = rng.integers(0, len(unique), size=(5000, len(unique)))
    boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {
        "score_before":float(old_score.mean()),
        "score_after":float(new_score.mean()),
        "delta":float(diff.mean()),
        "special_rows":int(special.sum()),
        "special_score_before":float(old_score[special].mean()),
        "special_score_after":float(new_score[special].mean()),
        "changed_rows":len(changed),
        "bootstrap_unit":bootstrap_unit,
        "bootstrap_95":list(map(float,np.quantile(boot,[0.025,0.975]))),
        "changed":changed,
    }


def main():
    features = json.loads((HERE / "features.json").read_text())
    with (HERE / "full_union_model_oof.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    with (HERE / "comparison_folds.csv").open(newline="") as file:
        folds = {row["filename"]:row for row in csv.DictReader(file)}
    assert len(rows) == 1497 and sum(trigger(f) for f in features.values()) == 7
    views = {}
    for alpha in (0.5, 0.75, 1.0, 1.25):
        for cap_reference in (True, False):
            key = f"alpha_{alpha:g}_capped_{int(cap_reference)}"
            views[key] = {split:evaluate(rows,features,folds,split,alpha,cap_reference)
                          for split in ("matched","structural")}
            print(key, "matched", round(views[key]["matched"]["score_after"],6),
                  "stress", round(views[key]["structural"]["score_after"],6))
    report = {"method":"nearest training circuit in log operations, within threshold; "
               "apply a max floor only to reset+multi-qubit+Grover-like circuits",
              "views":views}
    (HERE / "reset_family_floor_probe.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
