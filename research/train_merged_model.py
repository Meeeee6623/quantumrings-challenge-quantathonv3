"""Train and validate the merged Quantum Rings submission model.

This keeps the parallel research flow's bounded QASM parser and angle-aware
chi walk, then adds two techniques validated in the independent pipeline:

* one log-runtime ExtraTrees specialist for each guaranteed threshold; and
* a per-threshold timeout-risk classifier used as a conservative cap router.

The global and threshold-specialist regressors are averaged in log10 seconds,
which matches the challenge loss geometry.  Evaluation uses the committed,
label-free fixed folds so every threshold row from a circuit stays together.

Run from the repository root:

    uv run --locked python research/train_merged_model.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'quantathon-harness'), str(ROOT / 'research')]
from chi_walk import (  # noqa: E402
    BASIS_ROTATION_TOLERANCE,
    MODEL_FEATURE_NAMES,
    model_features,
    select_model_features,
)
from train_runtime import CAP, matrix, score  # noqa: E402

ARTIFACT = ROOT / 'quantathon-harness' / 'artifacts' / 'runtime_model.joblib'
REPORT = ROOT / 'research' / 'merged_model_validation.json'
OOF = ROOT / 'research' / 'merged_model_oof.csv'
THRESHOLDS = (16, 64, 512)
SPECIALIST_WEIGHT = 0.5
TIMEOUT_CUTOFF = 0.30


def global_regressor(seed=17):
    return ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=2,
        max_features=.9,
        n_jobs=-1,
        random_state=seed,
    )


def threshold_regressor(seed=17):
    return ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=1,
        max_features=1.0,
        n_jobs=-1,
        random_state=seed,
    )


def timeout_classifier(seed=17):
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=.8,
        class_weight='balanced',
        n_jobs=-1,
        random_state=seed,
    )


def load_training_table():
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    walks = json.loads((ROOT / 'research' / 'chi_walk_angle_cache.json').read_text())['tables']
    with (ROOT / 'runtime-data.csv').open(newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['filename'] in features]

    base = joblib.load(ROOT / 'research' / 'runtime_model_before_chi_walk.joblib')
    base_columns = base['columns']
    flat_walks = {name:model_features(walk) for name,walk in walks.items()}
    selected = [select_model_features(flat_walks[row['filename']],row['threshold'])
                for row in rows]
    walk_x = np.asarray([[entry[name] for name in MODEL_FEATURE_NAMES]
                         for entry in selected],dtype=float)
    X = np.column_stack((matrix(rows,features,base_columns),walk_x))
    columns = base_columns + list(MODEL_FEATURE_NAMES)
    y = np.asarray([math.log10(CAP if row['status'] == 'timeout'
                               else float(row['duration_s'])) for row in rows])
    timeout = np.asarray([row['status'] == 'timeout' for row in rows])
    thresholds = np.asarray([int(row['threshold']) for row in rows])
    names = np.asarray([row['filename'] for row in rows])
    return rows, X, columns, y, timeout, thresholds, names


def load_fold_table():
    with (ROOT / 'research' / 'comparison_folds.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    by_name = {row['filename']:row for row in rows}
    return by_name


def positive_probability(classifier, X):
    classes = list(classifier.classes_)
    if 1 not in classes:
        return np.zeros(len(X),dtype=float)
    return classifier.predict_proba(X)[:,classes.index(1)]


def cross_predict(X, y, timeout, thresholds, names, fold_by_name, split):
    fold_key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
    assigned = np.asarray([int(fold_by_name[name][fold_key]) for name in names])
    global_log = np.zeros(len(y),dtype=float)
    specialist_log = np.zeros(len(y),dtype=float)
    timeout_probability = np.zeros(len(y),dtype=float)

    for fold in sorted(set(assigned)):
        train = assigned != fold
        test = assigned == fold
        global_model = global_regressor(17)
        global_model.fit(X[train],y[train])
        global_log[test] = global_model.predict(X[test])

        for threshold in THRESHOLDS:
            train_t = train & (thresholds == threshold)
            test_t = test & (thresholds == threshold)
            if not test_t.any():
                continue
            specialist = threshold_regressor(17 + fold)
            specialist.fit(X[train_t],y[train_t])
            specialist_log[test_t] = specialist.predict(X[test_t])
            classifier = timeout_classifier(17 + fold)
            classifier.fit(X[train_t],timeout[train_t].astype(int))
            timeout_probability[test_t] = positive_probability(classifier,X[test_t])

    blended_log = ((1-SPECIALIST_WEIGHT)*global_log
                   + SPECIALIST_WEIGHT*specialist_log)
    merged_seconds = np.power(10.0,np.clip(blended_log,-9,9))
    merged_seconds = np.where(timeout_probability >= TIMEOUT_CUTOFF,CAP,merged_seconds)
    return {
        'global_log':global_log,
        'specialist_log':specialist_log,
        'timeout_probability':timeout_probability,
        'merged_seconds':merged_seconds,
    }


def metrics(y, timeout, thresholds, predicted_seconds):
    predicted_log = np.log10(np.maximum(1e-9,predicted_seconds))
    row_scores = score(y,predicted_log,timeout)
    actual = np.power(10.0,y)
    effective = np.where(timeout,np.minimum(predicted_seconds,CAP),predicted_seconds)
    factor = np.power(10.0,np.abs(np.log10(np.maximum(1e-9,effective)/actual)))
    return {
        'score':float(row_scores.mean()),
        'timeout_score':float(row_scores[timeout].mean()),
        'median_factor_error':float(np.median(factor)),
        'p95_factor_error':float(np.quantile(factor,.95)),
        'tenfold_misses':int((factor > 10).sum()),
        'predicted_timeouts':int((predicted_seconds == CAP).sum()),
        'threshold_scores':{
            str(threshold):float(row_scores[thresholds == threshold].mean())
            for threshold in THRESHOLDS
        },
    }


def paired_bootstrap(y, timeout, names, fold_by_name, split, current_seconds, merged_seconds):
    current = score(y,np.log10(np.maximum(1e-9,current_seconds)),timeout)
    merged = score(y,np.log10(np.maximum(1e-9,merged_seconds)),timeout)
    difference = merged-current
    if split == 'matched':
        units = names
        unit_name = 'circuit'
    else:
        units = np.asarray([fold_by_name[name]['structural_cluster'] for name in names])
        unit_name = 'structural_cluster'
    unique, group = np.unique(units,return_inverse=True)
    sums = np.bincount(group,weights=difference,minlength=len(unique))
    counts = np.bincount(group,minlength=len(unique))
    rng = np.random.default_rng(41)
    draws = rng.integers(0,len(unique),size=(5000,len(unique)))
    boot = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return {
        'unit':unit_name,
        'delta_merged_minus_current':float(difference.mean()),
        'delta_95':list(map(float,np.quantile(boot,[.025,.975]))),
    }


def fit_final_artifact(X, y, timeout, thresholds, columns):
    global_model = global_regressor(17)
    global_model.fit(X,y)
    specialists = {}
    classifiers = {}
    for threshold in THRESHOLDS:
        mask = thresholds == threshold
        regressor = threshold_regressor(17)
        regressor.fit(X[mask],y[mask])
        specialists[threshold] = regressor
        classifier = timeout_classifier(17)
        classifier.fit(X[mask],timeout[mask].astype(int))
        classifiers[threshold] = classifier
    artifact = {
        'artifact_version':'merged_threshold_experts_v1',
        'columns':columns,
        # Retain estimator for compatibility with older inspection scripts.
        'estimator':global_model,
        'global_estimator':global_model,
        'threshold_specialists':specialists,
        'threshold_specialist_weight':SPECIALIST_WEIGHT,
        'timeout_classifiers':classifiers,
        'timeout_probability_cutoff':TIMEOUT_CUTOFF,
        'view':'geometry_chi_angle_threshold_experts',
        'chi_walk_budget_s':3.0,
        'chi_walk_large_cutoff_bytes':40_000_000,
        'chi_walk_rotation_tolerance_rad':BASIS_ROTATION_TOLERANCE,
        'training_rows':int(len(y)),
        'known_thresholds':THRESHOLDS,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump(artifact,ARTIFACT,compress=3)


def main():
    rows, X, columns, y, timeout, thresholds, names = load_training_table()
    fold_by_name = load_fold_table()
    predictions = {}
    report = {
        'artifact_version':'merged_threshold_experts_v1',
        'rows':len(rows),
        'circuits':len(set(names)),
        'features':len(columns),
        'specialist_weight':SPECIALIST_WEIGHT,
        'timeout_probability_cutoff':TIMEOUT_CUTOFF,
        'splits':{},
    }
    for split in ('matched','structural'):
        pred = cross_predict(X,y,timeout,thresholds,names,fold_by_name,split)
        predictions[split] = pred
        current_seconds = np.power(10.0,pred['global_log'])
        split_report = {
            'current_global':metrics(y,timeout,thresholds,current_seconds),
            'merged':metrics(y,timeout,thresholds,pred['merged_seconds']),
            'paired_bootstrap':paired_bootstrap(
                y,timeout,names,fold_by_name,split,current_seconds,pred['merged_seconds']),
        }
        report['splits'][split] = split_report
        print(split,json.dumps(split_report,indent=2),flush=True)

    with OOF.open('w',newline='') as file:
        fields = ['filename','threshold','status','actual_s']
        for split in ('matched','structural'):
            fields += [f'{split}_global_pred_s',f'{split}_specialist_pred_s',
                       f'{split}_timeout_probability',f'{split}_merged_pred_s']
        writer = csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for i,row in enumerate(rows):
            entry = {
                'filename':row['filename'],
                'threshold':row['threshold'],
                'status':row['status'],
                'actual_s':10**y[i],
            }
            for split in ('matched','structural'):
                entry.update({
                    f'{split}_global_pred_s':10**predictions[split]['global_log'][i],
                    f'{split}_specialist_pred_s':10**predictions[split]['specialist_log'][i],
                    f'{split}_timeout_probability':predictions[split]['timeout_probability'][i],
                    f'{split}_merged_pred_s':predictions[split]['merged_seconds'][i],
                })
            writer.writerow(entry)

    fit_final_artifact(X,y,timeout,thresholds,columns)
    report['artifact'] = str(ARTIFACT.relative_to(ROOT))
    report['oof_predictions'] = str(OOF.relative_to(ROOT))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'artifact':str(ARTIFACT),'report':str(REPORT),'oof':str(OOF)},indent=2))


if __name__ == '__main__':
    main()
