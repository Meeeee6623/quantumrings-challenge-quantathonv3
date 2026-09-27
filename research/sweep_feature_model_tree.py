"""Run the same four-model sweep for each cumulative QASM feature pack.

Run: uv run --locked python research/sweep_feature_model_tree.py

The sweep is presentation evidence, not the production fit. It holds the
official matched circuit folds, target, categorical simulator setting, and
global-regressor hyperparameters fixed. It does not change the model artifact.
Results are checkpointed after each fit in presentation_figures/model_sweep.json.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "quantathon-harness"), str(ROOT / "research")]

from chi_walk import MODEL_FEATURE_NAMES, model_features, select_model_features  # noqa: E402
from evaluate_chi_randomness import NEW as CHI_CANDIDATES, INTERACTIONS, ESTIMATE  # noqa: E402
from evaluate_geometry_families import ADDED, FAMILY  # noqa: E402
from train_runtime import CAP, candidates, columns_for, matrix, score  # noqa: E402

OUT = ROOT / "research/presentation_figures/model_sweep.json"
PACKS = (
    ("basic", "Basic counts + depth"),
    ("normal", "Gate mix + temporal QASM"),
    ("chi_diversity", "+ χ bound + diversity"),
    ("geometry", "+ graph / cut geometry"),
    ("soft_patterns", "+ soft circuit patterns"),
    ("chi_walk", "+ χ walk"),
    ("angle_gated", "Angle-gated walk + angle stats"),
    ("hard_family", "External family probabilities"),
)
MODEL_NAMES = ("ridge", "hist_boost", "random_forest", "extra_trees")


def base_name(column: str) -> str:
    return column[4:] if column.startswith("log_") else column


def load_inputs():
    features = json.loads((ROOT / "research/features.json").read_text())
    with (ROOT / "runtime-data.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["filename"] in features]
    assert len(rows) == 1497 and len(features) == 532
    y = np.array([math.log10(CAP if row["status"] == "timeout"
                             else float(row["duration_s"])) for row in rows])
    timeout = np.array([row["status"] == "timeout" for row in rows])
    with (ROOT / "research/comparison_folds.csv").open(newline="") as handle:
        fold_by_name = {row["filename"]: int(row["matched_fold"])
                        for row in csv.DictReader(handle)}
    assigned = np.array([fold_by_name[row["filename"]] for row in rows])
    assert set(assigned) == set(range(5))

    old = json.loads((ROOT / "research/chi_walk_cache.json").read_text())["tables"]
    gated = json.loads((ROOT / "research/chi_walk_angle_cache.json").read_text())["tables"]
    old_flat = {name: model_features(walk) for name, walk in old.items()}
    gated_flat = {name: model_features(walk) for name, walk in gated.items()}
    old_matrix = np.array([[select_model_features(old_flat[row["filename"]],
                                                   row["threshold"])[name]
                            for name in MODEL_FEATURE_NAMES[:8]] for row in rows], dtype=float)
    gated_matrix = np.array([[select_model_features(gated_flat[row["filename"]],
                                                     row["threshold"])[name]
                              for name in MODEL_FEATURE_NAMES] for row in rows], dtype=float)
    assert old_matrix.shape == (1497, 8)
    assert gated_matrix.shape == (1497, 12)
    settings = np.array([[float(int(row["threshold"]) == t) for t in (16, 64, 512)]
                         for row in rows])
    family = json.loads((ROOT / "research/algorithm_geometry_predictions.json").read_text())
    family_names = sorted(next(iter(family.values())))
    family_matrix = np.array([[family[row["filename"]][name] for name in family_names]
                              + [max(family[row["filename"]].values())]
                              for row in rows], dtype=float)
    assert family_matrix.shape == (1497, 9)
    return (features, rows, y, timeout, assigned, settings, old_matrix,
            gated_matrix, family_matrix, family_names)


def feature_matrix(pack, features, rows, settings, old_walk, gated_walk,
                   family_matrix, family_names):
    if pack == "basic":
        columns = columns_for(features, "basic")
    else:
        columns = columns_for(features, "all")
    blocked = {
        "basic": set(),
        "normal": CHI_CANDIDATES | ADDED,
        "chi_diversity": INTERACTIONS | ESTIMATE | ADDED,
        "geometry": INTERACTIONS | ESTIMATE | FAMILY,
        "soft_patterns": INTERACTIONS | ESTIMATE,
        "chi_walk": INTERACTIONS | ESTIMATE,
        "angle_gated": INTERACTIONS | ESTIMATE,
        "hard_family": INTERACTIONS | ESTIMATE,
    }[pack]
    columns = [name for name in columns
               if name not in ("threshold", "log_threshold")
               and base_name(name) not in blocked]
    X = np.column_stack((matrix(rows, features, columns), settings))
    columns += [f"setting_{t}" for t in (16, 64, 512)]
    if pack == "chi_walk":
        X = np.column_stack((X, old_walk))
        columns += list(MODEL_FEATURE_NAMES[:8])
    if pack in ("angle_gated", "hard_family"):
        X = np.column_stack((X, gated_walk))
        columns += list(MODEL_FEATURE_NAMES)
    if pack == "hard_family":
        X = np.column_stack((X, family_matrix))
        columns += [f"external_family_{name}" for name in family_names]
        columns += ["external_family_confidence"]
    assert np.isfinite(X).all(), pack
    assert X.shape[1] == len(columns)
    return X, columns


def main():
    (features, rows, y, timeout, assigned, settings, old_walk, gated_walk,
     family_matrix, family_names) = load_inputs()
    if OUT.exists():
        result = json.loads(OUT.read_text())
        assert result["model_names"] == list(MODEL_NAMES)
        names = [p[0] for p in PACKS]
        assert result["pack_names"] == names[:len(result["pack_names"])]
        result["pack_names"] = names
    else:
        result = {"rows": len(rows), "circuits": len(features),
                  "folds": 5, "fold_source": "research/comparison_folds.csv:matched_fold",
                  "setting_encoding": "one_hot_categorical", "target": "log10(runtime_seconds)",
                  "model_names": list(MODEL_NAMES),
                  "pack_names": [p[0] for p in PACKS],
                  "packs": {}}
    for pack, label in PACKS:
        X, columns = feature_matrix(pack, features, rows, settings, old_walk,
                                    gated_walk, family_matrix, family_names)
        record = result["packs"].setdefault(pack, {"label": label,
                                                    "columns": columns, "models": {}})
        record["label"] = label
        assert record["columns"] == columns
        for model_name in MODEL_NAMES:
            if model_name in record["models"]:
                print(f"cached {pack}/{model_name}", flush=True)
                continue
            started = time.perf_counter()
            pred = np.zeros(len(rows))
            for fold in range(5):
                train, test = assigned != fold, assigned == fold
                estimator = candidates()[model_name]()
                if hasattr(estimator, "n_jobs"):
                    estimator.n_jobs = 4
                estimator.fit(X[train], y[train])
                pred[test] = estimator.predict(X[test])
            s = score(y, pred, timeout)
            record["models"][model_name] = {
                "score": float(s.mean()),
                "timeout_score": float(s[timeout].mean()),
                "seconds": round(time.perf_counter()-started, 2),
            }
            OUT.write_text(json.dumps(result, indent=2) + "\n")
            print(f"{pack}/{model_name}: {s.mean():.6f} "
                  f"({record['models'][model_name]['seconds']:.1f}s)", flush=True)
    print(f"Complete {len(PACKS)*len(MODEL_NAMES)} grouped fits: {OUT}")


if __name__ == "__main__":
    main()
