"""Build reviewed, source-linked evidence for the local runtime report."""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def read_csv(name):
    with (RESEARCH / name).open(newline="") as file:
        return list(csv.DictReader(file))


def source(label, files, steps, definitions=None):
    return {
        "label": label,
        "files": files,
        "evidenceFlow": [{"title": "Local reproducible run", "detail": steps}],
        "metricDefinitions": definitions or [],
    }


def query(rows, label, files, steps, definitions=None):
    return {"rows": rows, "source": source(label, files, steps, definitions)}


def main():
    validation = json.loads((RESEARCH / "validation.json").read_text())
    features = json.loads((RESEARCH / "features.json").read_text())
    oof = read_csv("oof_predictions.csv")
    harness = read_csv("training_submission.csv")
    assert len(oof) == 1497 and len(harness) == 1596

    score_rows = []
    by_circuit = defaultdict(list)
    bins = {"≤1.25×": 0, "1.25–2×": 0, "2–10×": 0, ">10×": 0}
    for row in oof:
        actual, pred, score = (float(row[k]) for k in ("actual_s", "predicted_s", "score"))
        factor = max(actual / pred, pred / actual)
        if factor <= 1.25:
            bins["≤1.25×"] += 1
        elif factor <= 2:
            bins["1.25–2×"] += 1
        elif factor <= 10:
            bins["2–10×"] += 1
        else:
            bins[">10×"] += 1
        score_rows.append({
            "filename": row["filename"], "threshold": row["threshold"],
            "status": row["status"], "actual_s": actual, "predicted_s": pred,
            "actual_log10_s": math.log10(actual), "predicted_log10_s": math.log10(pred),
            "score": score, "factor_error": factor,
        })
        by_circuit[row["filename"]].append(score)

    # Resample circuits, preserving the three-threshold rows within each draw.
    import random
    rng = random.Random(17)
    circuit_ids = sorted(by_circuit)
    samples = []
    for _ in range(1500):
        drawn = rng.choices(circuit_ids, k=len(circuit_ids))
        scores = [s for circuit in drawn for s in by_circuit[circuit]]
        samples.append(sum(scores) / len(scores))
    samples.sort()
    lo, hi = samples[37], samples[1462]

    model_rows = [{
        "features": view, "scorePct": 100 * next(r["score"] for r in validation["results"]
                                             if r["model"] == "extra_trees" and r["view"] == view),
    } for view in ("basic", "raw", "all")]
    threshold_rows = [{"threshold": k, "scorePct": 100 * v}
                      for k, v in validation["winner"]["threshold_scores"].items()]
    error_rows = [{"factorBand": k, "cases": v} for k, v in bins.items()]
    stress_rows = [{"fold": f"Fold {i}", "scorePct": 100 * v}
                   for i, v in enumerate(validation["structural_cluster_fold_scores"], 1)]

    parse_by_circuit = {}
    pred_times = []
    for row in harness:
        parse_by_circuit[row["filename"]] = float(row["parse_s"])
        pred_times.append(float(row["predict_s"]))
    timing_rows = [{
        "filename": name,
        "qasmMB": float(features[name]["qasm_bytes"]) / 1e6,
        "parse_s": parse_s,
    } for name, parse_s in parse_by_circuit.items()]
    all_predictions = [float(row["pred_duration_s"]) for row in harness]
    assert len(parse_by_circuit) == 532 and all(math.isfinite(v) and v > 0 for v in all_predictions)
    cap_exceeded = sum(v > 15 for v in parse_by_circuit.values()) + sum(v > 15 for v in pred_times)
    summary = [{
        "groupedScorePct": 100 * validation["winner"]["score"],
        "groupedBootLoPct": 100 * lo, "groupedBootHiPct": 100 * hi,
        "structuralScorePct": 100 * validation["structural_cluster_holdout_score"],
        "labeledRows": len(oof), "circuits": len(parse_by_circuit),
        "harnessRows": len(harness), "timeoutRows": sum(r["status"] == "timeout" for r in oof),
        "zeroScoreRows": sum(r["score"] == 0 for r in score_rows),
        "maxParseS": max(parse_by_circuit.values()),
        "maxPredictS": max(pred_times), "capExceeded": cap_exceeded,
    }]
    snapshot = {
        "title": "Runtime predictions hold up on known circuit patterns; new shapes remain difficult",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "reviewed", "filters": [], "report": {},
        "queries": {
            "summary": query(summary, "Run and validation summary",
                ["research/validation.json", "research/oof_predictions.csv", "research/training_submission.csv"],
                "Run `uv run --locked python research/train_runtime.py --train-only`, then `uv run --locked python quantathon-harness/run.py --team quantum-rings --circuits training_circuits --out research/training_submission.csv`; aggregate the emitted rows in research/make_report_snapshot.py."),
            "models": query(model_rows, "Five-fold circuit-grouped cross-validation by feature set",
                ["research/validation.json"],
                "Select the ExtraTrees results for basic, raw, and all features from validation.json; multiply mean challenge score by 100."),
            "thresholds": query(threshold_rows, "Held-out score by threshold",
                ["research/validation.json"],
                "Read the winner threshold_scores in validation.json; multiply by 100."),
            "oof": query(score_rows, "Out-of-fold circuit predictions",
                ["research/oof_predictions.csv"],
                "Each circuit's labeled threshold rows are held out together in five-fold GroupKFold. Timeout labels use the 14,400-second cap. Log columns are base-10 logarithms of the actual and predicted durations in seconds."),
            "errors": query(error_rows, "Out-of-fold factor-error distribution",
                ["research/oof_predictions.csv"],
                "Compute max(actual/predicted, predicted/actual) from each out-of-fold prediction, then count disjoint factor-error bins."),
            "stress": query(stress_rows, "Structural-cluster holdout fold scores",
                ["research/validation.json"],
                "Use structural_cluster_fold_scores from validation.json; clusters use permitted circuit features, with GroupKFold holding coarse clusters out."),
            "timing": query(timing_rows, "Full harness parser timings",
                ["research/training_submission.csv", "research/features.json"],
                "Take one parser time per circuit from the full harness output and join by filename to extracted QASM bytes; divide bytes by 1e6."),
        },
    }
    score_definition = {
        "label": "Challenge duration score",
        "definition": "Mean of per-row max(0, 1 - abs(log10(predicted seconds / actual seconds)) / 2); timeout actual and predicted seconds are capped at 14,400 as in score.py.",
        "formula": "mean(max(0, 1 - abs(log10(predicted_s / actual_s)) / 2))",
        "sourceLineage": [{"files": ["quantathon-harness/score.py"]}],
    }
    for key in ("summary", "models", "thresholds", "oof", "stress"):
        snapshot["queries"][key]["source"]["metricDefinitions"].append(score_definition)
    snapshot["queries"]["errors"]["source"]["metricDefinitions"].append({
        "label": "Factor error", "definition": "Larger of actual divided by predicted and predicted divided by actual; bins are disjoint and count labeled circuit-threshold pairs.",
        "formula": "max(actual_s / predicted_s, predicted_s / actual_s)",
        "sourceLineage": [{"files": ["research/oof_predictions.csv"]}],
    })
    snapshot["queries"]["timing"]["source"]["metricDefinitions"].append({
        "label": "Parser time", "definition": "Seconds spent in RuntimeModel.featurize for one circuit in the supplied harness, excluding QASM decompression.",
        "sourceLineage": [{"files": ["research/training_submission.csv"]}],
    })
    (RESEARCH / "report_snapshot.json").write_text(json.dumps(snapshot, separators=(",", ":")))
    (RESEARCH / "report_metrics.json").write_text(json.dumps(summary[0], indent=2) + "\n")
    print(json.dumps(summary[0], indent=2))
    print("error bins", bins)


if __name__ == "__main__":
    main()
