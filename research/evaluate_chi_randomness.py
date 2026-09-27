"""Ablate cut-rank upper bounds and circuit-randomness proxies.

Keeps the final submission artifact untouched. Run after refreshing features.json:
    uv run --locked python research/train_runtime.py --extract-only
    uv run --locked python research/evaluate_chi_randomness.py
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_runtime import ROOT, CAP, candidates, columns_for, matrix, score

CHI = {
    "chi_upper_peak", "chi_upper_mid", "chi_upper_mean",
    "chi_capacity_fraction", "chi_unsaturated_fraction", "chi_mid_budget",
}
RANDOM = {
    "random_gate_entropy", "random_bigram_entropy", "random_angle_entropy",
    "random_pair_entropy", "random_window_entropy", "randomness_proxy",
}
INTERACTIONS = {"chi_random_peak", "chi_random_pressure"}
ESTIMATE = {"chi_est_log2_peak", "chi_est_log2_mid", "chi_est_log2_mean",
            "chi_est_capacity_fraction"}
NEW = CHI | RANDOM | INTERACTIONS | ESTIMATE
VIEWS = {
    "baseline": set(),
    "chi_only": CHI,
    "random_only": RANDOM,
    "chi_plus_random": CHI | RANDOM,
    "effective_chi_only": ESTIMATE,
    "chi_plus_effective_peak": CHI | RANDOM | {"chi_est_log2_peak", "chi_est_capacity_fraction"},
    "chi_plus_random_effective": CHI | RANDOM | ESTIMATE,
    "chi_random_interactions": CHI | RANDOM | INTERACTIONS,
}


def boot_delta(a, b, names, replicates=1500):
    """Resample circuits, retaining their threshold rows in each draw."""
    import random
    by_name = {}
    for i, name in enumerate(names):
        by_name.setdefault(name, []).append(i)
    circuits = sorted(by_name)
    rng = random.Random(23)
    diffs = []
    for _ in range(replicates):
        drawn = rng.choices(circuits, k=len(circuits))
        indexes = [i for circuit in drawn for i in by_name[circuit]]
        diffs.append(float(np.mean((b-a)[indexes])))
    diffs.sort()
    return [diffs[int(.025*replicates)], diffs[int(.975*replicates)-1]]


def main():
    feats = json.loads((ROOT / "research/features.json").read_text())
    assert len(feats) == 532
    assert all(NEW <= set(f) for f in feats.values())
    with (ROOT / "runtime-data.csv").open(newline="") as file:
        rows = [r for r in csv.DictReader(file) if r["filename"] in feats]
    names = np.array([r["filename"] for r in rows])
    y = np.array([math.log10(CAP if r["status"] == "timeout" else float(r["duration_s"])) for r in rows])
    timeout = np.array([r["status"] == "timeout" for r in rows])

    # Hash exactly the pre-ablation feature profile to preserve the original
    # circuit-grouped fold assignment and keep paired differences meaningful.
    signatures = {name: hashlib.sha1(json.dumps({k:v for k,v in f.items() if k not in NEW},
                                               sort_keys=True).encode()).hexdigest()
                  for name,f in feats.items()}
    groups = np.array([signatures[name] for name in names])
    normal_splits = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    circuit_names = sorted(feats)
    cluster_keys = ("n_qubits", "ops", "two_q_ratio", "nonclifford_ratio",
                    "conditional", "custom_definitions", "qasm_bytes")
    C = np.array([[math.log1p(max(0,float(feats[name].get(k,0))))
                   for k in cluster_keys] for name in circuit_names])
    cluster_ids = KMeans(n_clusters=12, n_init=10, random_state=17).fit_predict(StandardScaler().fit_transform(C))
    cluster_lookup = dict(zip(circuit_names, cluster_ids))
    stress_groups = np.array([cluster_lookup[name] for name in names])
    stress_splits = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, stress_groups))

    all_columns = columns_for(feats, "all")
    results = []
    score_vectors = {}
    structural_score_vectors = {}
    prediction_vectors = {}
    for view, extra in VIEWS.items():
        cols = [c for c in all_columns if (c[4:] if c.startswith("log_") and c != "log_threshold" else c) not in NEW-extra]
        X = matrix(rows, feats, cols)
        predictions = {}
        per_split = {}
        for split_name, splits in (("circuit", normal_splits), ("structural", stress_splits)):
            pred = np.zeros(len(rows))
            fold_scores = []
            for train_idx, test_idx in splits:
                est = candidates()["extra_trees"]()
                est.fit(X[train_idx], y[train_idx])
                pred[test_idx] = est.predict(X[test_idx])
                fold_scores.append(float(score(y[test_idx], pred[test_idx], timeout[test_idx]).mean()))
            s = score(y,pred,timeout)
            per_split[split_name] = pred
            predictions[split_name] = {"score":float(s.mean()), "fold_scores":fold_scores,
                                       "timeout_score":float(s[timeout].mean())}
            if split_name == "circuit":
                score_vectors[view] = s
            else:
                structural_score_vectors[view] = s
        row = {"view":view, "new_features":sorted(extra), "columns":len(cols), **predictions}
        results.append(row)
        prediction_vectors[view] = per_split
        print(json.dumps(row), flush=True)

    # The effective-χ proxy may help novel structures while the additive
    # feature model remains more reliable on ordinary grouped folds.
    blends = []
    for target in ("chi_plus_effective_peak", "chi_random_interactions"):
        for weight in (.25, .5, .75):
            parts = {}
            for split_name in ("circuit", "structural"):
                pred = ((1-weight)*prediction_vectors["chi_plus_random"][split_name]
                        + weight*prediction_vectors[target][split_name])
                s = score(y,pred,timeout)
                parts[split_name] = {"score":float(s.mean()), "timeout_score":float(s[timeout].mean())}
            result = {"baseline_view":"chi_plus_random", "target_view":target,
                      "target_weight":weight, **parts}
            blends.append(result)
            print(json.dumps(result),flush=True)

    comparisons = []
    base = score_vectors["baseline"]
    for row in results[1:]:
        view = row["view"]
        comparisons.append({
            "view":view,
            "grouped_delta":float(np.mean(score_vectors[view]-base)),
            "grouped_delta_bootstrap_95":boot_delta(base,score_vectors[view],names),
            "structural_delta":row["structural"]["score"]-results[0]["structural"]["score"],
            "structural_delta_bootstrap_95":boot_delta(
                structural_score_vectors["baseline"],structural_score_vectors[view],names),
        })
    report = {"rows":len(rows), "circuits":len(set(names)), "folds":5,
              "model":"extra_trees", "results":results, "comparisons":comparisons,
              "blends":blends,
              "bound":"For qubit-order cut k, log2(chi) <= min(k, n-k, sum of operator-Schmidt log-rank bounds for gates crossing k). The bound assumes product-state input and counts all gates before possible cancellation.",
              "effective_chi":"log2(chi_est) = randomness_proxy * log2(chi_upper), so the heuristic rank approaches the cut upper bound as randomness approaches 1.",
              "scope":"Exploratory feature ablation on the supplied training circuits; no hidden-circuit score or simulator timing claim."}
    target = ROOT / "research/chi_randomness_ablation.json"
    target.write_text(json.dumps(report,indent=2)+"\n")
    oof = ROOT / "research/chi_randomness_oof.csv"
    with oof.open("w",newline="") as file:
        writer = csv.writer(file)
        views = list(VIEWS)
        writer.writerow(["filename","threshold","actual_s","status","chi_upper_peak",
                         "chi_capacity_fraction","randomness_proxy","chi_random_pressure"]
                        +[f"{v}_{split}_pred_s" for split in ("circuit","structural") for v in views])
        for i,row in enumerate(rows):
            f = feats[row["filename"]]
            writer.writerow([row["filename"],row["threshold"],10**y[i],row["status"],
                             f["chi_upper_peak"],f["chi_capacity_fraction"],
                             f["randomness_proxy"],f["chi_random_pressure"]]
                            +[10**prediction_vectors[v][split][i]
                              for split in ("circuit","structural") for v in views])
    print("WROTE",target,flush=True)
    print("WROTE",oof,flush=True)


if __name__ == "__main__":
    main()
