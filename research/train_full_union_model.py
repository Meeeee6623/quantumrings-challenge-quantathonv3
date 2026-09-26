"""Train the primary dual-parser union model.

The global component uses the 243-column bounded geometry/chi representation.
Threshold specialists and timeout classifiers additionally see the independent
237-column structural/DAG/angle representation.  The two namespaces are kept
separate, even where concepts overlap, so the experiment is reproducible.

Run:

    uv run --locked python research/extract_extended_features.py  # once
    uv run --locked python research/train_full_union_model.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'quantathon-harness'),str(ROOT / 'research')]
from chi_walk import BASIS_ROTATION_TOLERANCE  # noqa: E402
from runtime_floors import apply_floor, eligible, large_work_floor  # noqa: E402
from template_analogues import blend_with_analogues, signature  # noqa: E402
from train_merged_model import (  # noqa: E402
    ARTIFACT,
    CAP,
    THRESHOLDS,
    global_regressor,
    load_fold_table,
    load_training_table,
    metrics,
    paired_bootstrap,
    positive_probability,
    threshold_regressor,
    timeout_classifier,
)

EXTENDED = ROOT / 'research' / 'extended_features.json'
REPORT = ROOT / 'research' / 'full_union_model_validation.json'
OOF = ROOT / 'research' / 'full_union_model_oof.csv'
SPECIALIST_WEIGHT = 0.5
TIMEOUT_CUTOFF = 0.35


def reset_family_references(rows, features, selected):
    return [(int(rows[i]['threshold']), float(features[rows[i]['filename']]['ops']),
             min(CAP, CAP if rows[i]['status'] == 'timeout'
                 else float(rows[i].get('duration_s',rows[i].get('actual_s')))))
            for i in range(len(rows)) if selected[i]
            and eligible(features[rows[i]['filename']])]


def floor_fold_predictions(predicted_seconds, rows, features, assigned):
    adjusted = predicted_seconds.copy()
    for fold in sorted(set(assigned)):
        train = assigned != fold
        references = reset_family_references(rows,features,train)
        for i in np.flatnonzero(assigned == fold):
            circuit_features = features[rows[i]['filename']]
            if eligible(circuit_features):
                adjusted[i] = apply_floor(adjusted[i],circuit_features,
                                          rows[i]['threshold'],references)
    return adjusted


def large_work_predictions(predicted_seconds, rows, features):
    return np.asarray([max(predicted_seconds[i],
                           large_work_floor(features[row['filename']]))
                       for i,row in enumerate(rows)])


def build_template_bank(rows, features, selected):
    bank = {}
    for i,row in enumerate(rows):
        if not selected[i]:
            continue
        key = (int(row['threshold']),*signature(features[row['filename']]))
        seconds = (CAP if row['status'] == 'timeout'
                   else float(row.get('duration_s',row.get('actual_s'))))
        bank.setdefault(key,[]).append(math.log10(seconds))
    return {key:tuple(values) for key,values in bank.items()}


def template_fold_predictions(predicted_seconds, rows, features, assigned):
    adjusted = predicted_seconds.copy()
    for fold in sorted(set(assigned)):
        bank = build_template_bank(rows,features,assigned != fold)
        for i in np.flatnonzero(assigned == fold):
            adjusted[i] = blend_with_analogues(adjusted[i],
                features[rows[i]['filename']],rows[i]['threshold'],bank)
    return adjusted


def load_extended_matrix(rows, thresholds):
    if not EXTENDED.exists():
        raise SystemExit('missing research/extended_features.json; run '
                         'research/extract_extended_features.py first')
    payload = json.loads(EXTENDED.read_text())
    columns = payload['columns']
    tables = payload['tables']
    matrix = np.empty((len(rows),len(columns)),dtype=float)
    for i,row in enumerate(rows):
        table = tables[row['filename']]
        for j,column in enumerate(columns):
            value = thresholds[i] if column == 'threshold' else table.get(column)
            matrix[i,j] = math.nan if value is None else float(value)
    return matrix,['extended__'+column for column in columns]


def cross_predict(parallel_X, union_X, y, timeout, thresholds, names, rows,
                  features, fold_by_name, split):
    fold_key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
    assigned = np.asarray([int(fold_by_name[name][fold_key]) for name in names])
    global_log = np.zeros(len(y),dtype=float)
    specialist_log = np.zeros(len(y),dtype=float)
    timeout_probability = np.zeros(len(y),dtype=float)
    for fold in sorted(set(assigned)):
        train = assigned != fold
        test = assigned == fold
        global_model = global_regressor(17)
        global_model.fit(parallel_X[train],y[train])
        global_log[test] = global_model.predict(parallel_X[test])
        for threshold in THRESHOLDS:
            train_t = train & (thresholds == threshold)
            test_t = test & (thresholds == threshold)
            if not test_t.any():
                continue
            specialist = threshold_regressor(17+fold)
            specialist.fit(union_X[train_t],y[train_t])
            specialist_log[test_t] = specialist.predict(union_X[test_t])
            classifier = timeout_classifier(17+fold)
            classifier.fit(union_X[train_t],timeout[train_t].astype(int))
            timeout_probability[test_t] = positive_probability(classifier,union_X[test_t])
    blended_log = ((1-SPECIALIST_WEIGHT)*global_log
                   + SPECIALIST_WEIGHT*specialist_log)
    seconds = np.power(10.0,np.clip(blended_log,-9,9))
    seconds = np.where(timeout_probability >= TIMEOUT_CUTOFF,CAP,seconds)
    adjusted = floor_fold_predictions(seconds,rows,features,assigned)
    large_adjusted = large_work_predictions(adjusted,rows,features)
    template_adjusted = template_fold_predictions(large_adjusted,rows,features,assigned)
    return {
        'global_log':global_log,
        'specialist_log':specialist_log,
        'timeout_probability':timeout_probability,
        'merged_seconds':seconds,
        'adjusted_seconds':adjusted,
        'large_adjusted_seconds':large_adjusted,
        'template_adjusted_seconds':template_adjusted,
    }


def fit_artifact(parallel_X, union_X, parallel_columns, union_columns,
                 y, timeout, thresholds, interval_log10_radius, references,
                 template_bank):
    global_model = global_regressor(17)
    global_model.fit(parallel_X,y)
    specialists = {}
    classifiers = {}
    for threshold in THRESHOLDS:
        mask = thresholds == threshold
        specialist = threshold_regressor(17)
        specialist.fit(union_X[mask],y[mask])
        specialists[threshold] = specialist
        classifier = timeout_classifier(17)
        classifier.fit(union_X[mask],timeout[mask].astype(int))
        classifiers[threshold] = classifier
    artifact = {
        'artifact_version':'full_union_threshold_experts_v4',
        'columns':union_columns,
        'global_columns':parallel_columns,
        'specialist_columns':union_columns,
        'classifier_columns':union_columns,
        'estimator':global_model,
        'global_estimator':global_model,
        'threshold_specialists':specialists,
        'threshold_specialist_weight':SPECIALIST_WEIGHT,
        'timeout_classifiers':classifiers,
        'timeout_probability_cutoff':TIMEOUT_CUTOFF,
        'reset_family_references':references,
        'template_analogue_bank':template_bank,
        'view':'full_union_geometry_chi_structural_dag_angle_threshold_experts',
        'chi_walk_budget_s':3.0,
        'chi_walk_large_cutoff_bytes':40_000_000,
        'chi_walk_rotation_tolerance_rad':BASIS_ROTATION_TOLERANCE,
        'training_rows':int(len(y)),
        'known_thresholds':THRESHOLDS,
        'interval_alpha':.10,
        'interval_log10_radius':float(interval_log10_radius),
    }
    joblib.dump(artifact,ARTIFACT,compress=3)


def main():
    rows, parallel_X, parallel_columns, y, timeout, thresholds, names = load_training_table()
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    extended_X, extended_columns = load_extended_matrix(rows,thresholds)
    union_X = np.column_stack((parallel_X,extended_X))
    union_columns = parallel_columns+extended_columns
    fold_by_name = load_fold_table()
    predictions = {}
    report = {
        'artifact_version':'full_union_threshold_experts_v4',
        'rows':len(rows),
        'circuits':len(set(names)),
        'parallel_features':len(parallel_columns),
        'extended_features':len(extended_columns),
        'union_features':len(union_columns),
        'specialist_weight':SPECIALIST_WEIGHT,
        'timeout_probability_cutoff':TIMEOUT_CUTOFF,
        'splits':{},
    }
    for split in ('matched','structural'):
        pred = cross_predict(parallel_X,union_X,y,timeout,thresholds,names,rows,
                             features,fold_by_name,split)
        predictions[split] = pred
        current_seconds = np.power(10.0,pred['global_log'])
        split_report = {
            'parallel_global':metrics(y,timeout,thresholds,current_seconds),
            'full_union':metrics(y,timeout,thresholds,pred['merged_seconds']),
            'reset_family_floor':metrics(y,timeout,thresholds,pred['adjusted_seconds']),
            'both_floors':metrics(y,timeout,thresholds,pred['large_adjusted_seconds']),
            'template_analogues':metrics(y,timeout,thresholds,pred['template_adjusted_seconds']),
            'template_vs_both_floors':paired_bootstrap(
                y,timeout,names,fold_by_name,split,pred['large_adjusted_seconds'],
                pred['template_adjusted_seconds']),
            'both_floors_vs_full_union':paired_bootstrap(
                y,timeout,names,fold_by_name,split,pred['merged_seconds'],
                pred['large_adjusted_seconds']),
            'large_floor_vs_reset_floor':paired_bootstrap(
                y,timeout,names,fold_by_name,split,pred['adjusted_seconds'],
                pred['large_adjusted_seconds']),
            'reset_floor_vs_full_union':paired_bootstrap(
                y,timeout,names,fold_by_name,split,pred['merged_seconds'],
                pred['adjusted_seconds']),
            'paired_bootstrap':paired_bootstrap(
                y,timeout,names,fold_by_name,split,current_seconds,
                pred['template_adjusted_seconds']),
        }
        report['splits'][split] = split_report
        print(split,json.dumps(split_report,indent=2),flush=True)

    with OOF.open('w',newline='') as file:
        fields = ['filename','threshold','status','actual_s']
        for split in ('matched','structural'):
            fields += [f'{split}_global_pred_s',f'{split}_specialist_pred_s',
                       f'{split}_timeout_probability',f'{split}_full_union_pred_s',
                       f'{split}_reset_floor_pred_s',f'{split}_both_floors_pred_s',
                       f'{split}_template_pred_s']
        writer = csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for i,row in enumerate(rows):
            entry = {'filename':row['filename'],'threshold':row['threshold'],
                     'status':row['status'],'actual_s':10**y[i]}
            for split in ('matched','structural'):
                entry.update({
                    f'{split}_global_pred_s':10**predictions[split]['global_log'][i],
                    f'{split}_specialist_pred_s':10**predictions[split]['specialist_log'][i],
                    f'{split}_timeout_probability':predictions[split]['timeout_probability'][i],
                    f'{split}_full_union_pred_s':predictions[split]['merged_seconds'][i],
                    f'{split}_reset_floor_pred_s':predictions[split]['adjusted_seconds'][i],
                    f'{split}_both_floors_pred_s':predictions[split]['large_adjusted_seconds'][i],
                    f'{split}_template_pred_s':predictions[split]['template_adjusted_seconds'][i],
                })
            writer.writerow(entry)
    matched_seconds = predictions['matched']['template_adjusted_seconds']
    interval_radius = float(np.quantile(
        np.abs(np.log10(np.maximum(1e-9,matched_seconds))-y),.90))
    report['uncertainty'] = {
        'method':'fixed-fold conformal-style absolute log10 residual radius',
        'nominal_coverage':.90,
        'log10_radius':interval_radius,
        'multiplicative_factor':float(10**interval_radius),
        'observed_oof_coverage':float(np.mean(
            np.abs(np.log10(np.maximum(1e-9,matched_seconds))-y) <= interval_radius)),
    }
    fit_artifact(parallel_X,union_X,parallel_columns,union_columns,
                 y,timeout,thresholds,interval_radius,
                 reset_family_references(rows,features,np.ones(len(rows),dtype=bool)),
                 build_template_bank(rows,features,np.ones(len(rows),dtype=bool)))
    report['artifact'] = str(ARTIFACT.relative_to(ROOT))
    report['oof_predictions'] = str(OOF.relative_to(ROOT))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'artifact':str(ARTIFACT),'report':str(REPORT),'oof':str(OOF)},indent=2))


if __name__ == '__main__':
    main()
