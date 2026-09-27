"""Train the pruned dual-parser union model.

The two parsers produce 480 candidate columns. Constant and exactly duplicate
columns are removed, then each model selects a smaller set by importance.
Validation learns those importance rankings inside each training fold.

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
from chi_walk import BASIS_ROTATION_TOLERANCE, model_features  # noqa: E402
from feature_pruning import importance_choice, redundant_columns  # noqa: E402
from rotation_calibration import calibrate_near_basis_runtime  # noqa: E402
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
GLOBAL_LIMIT = 120
SPECIALIST_LIMIT = 200
CLASSIFIER_LIMIT = 80


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


def near_basis_predictions(predicted_seconds, rows, features):
    return np.asarray([calibrate_near_basis_runtime(
        predicted_seconds[i],features[row['filename']],row['threshold'])
        for i,row in enumerate(rows)])


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
        global_indices = importance_choice(global_regressor(17),parallel_X[train],
                                           y[train],GLOBAL_LIMIT)
        global_model = global_regressor(17)
        global_model.fit(parallel_X[train][:,global_indices],y[train])
        global_log[test] = global_model.predict(parallel_X[test][:,global_indices])
        for threshold in THRESHOLDS:
            train_t = train & (thresholds == threshold)
            test_t = test & (thresholds == threshold)
            if not test_t.any():
                continue
            specialist_indices = importance_choice(
                threshold_regressor(17+fold),union_X[train_t],y[train_t],
                SPECIALIST_LIMIT)
            specialist = threshold_regressor(17+fold)
            specialist.fit(union_X[train_t][:,specialist_indices],y[train_t])
            specialist_log[test_t] = specialist.predict(
                union_X[test_t][:,specialist_indices])
            classifier_indices = importance_choice(
                timeout_classifier(17+fold),union_X[train_t],
                timeout[train_t].astype(int),CLASSIFIER_LIMIT)
            classifier = timeout_classifier(17+fold)
            classifier.fit(union_X[train_t][:,classifier_indices],
                           timeout[train_t].astype(int))
            timeout_probability[test_t] = positive_probability(
                classifier,union_X[test_t][:,classifier_indices])
    blended_log = ((1-SPECIALIST_WEIGHT)*global_log
                   + SPECIALIST_WEIGHT*specialist_log)
    seconds = np.power(10.0,np.clip(blended_log,-9,9))
    seconds = np.where(timeout_probability >= TIMEOUT_CUTOFF,CAP,seconds)
    adjusted = floor_fold_predictions(seconds,rows,features,assigned)
    large_adjusted = large_work_predictions(adjusted,rows,features)
    template_adjusted = template_fold_predictions(large_adjusted,rows,features,assigned)
    basis_adjusted = near_basis_predictions(template_adjusted,rows,features)
    return {
        'global_log':global_log,
        'specialist_log':specialist_log,
        'timeout_probability':timeout_probability,
        'merged_seconds':seconds,
        'adjusted_seconds':adjusted,
        'large_adjusted_seconds':large_adjusted,
        'template_adjusted_seconds':template_adjusted,
        'basis_adjusted_seconds':basis_adjusted,
    }


def fit_artifact(parallel_X, union_X, parallel_columns, union_columns,
                 y, timeout, thresholds, interval_log10_radius, references,
                 template_bank, artifact_version='pruned_union_threshold_experts_v7'):
    global_indices = importance_choice(global_regressor(17),parallel_X,y,GLOBAL_LIMIT)
    global_columns = [parallel_columns[j] for j in global_indices]
    global_model = global_regressor(17)
    global_model.fit(parallel_X[:,global_indices],y)
    specialists = {}
    classifiers = {}
    specialist_columns = {}
    classifier_columns = {}
    for threshold in THRESHOLDS:
        mask = thresholds == threshold
        specialist_indices = importance_choice(
            threshold_regressor(17),union_X[mask],y[mask],SPECIALIST_LIMIT)
        specialist_columns[threshold] = [union_columns[j] for j in specialist_indices]
        specialist = threshold_regressor(17)
        specialist.fit(union_X[mask][:,specialist_indices],y[mask])
        specialists[threshold] = specialist
        classifier_indices = importance_choice(
            timeout_classifier(17),union_X[mask],timeout[mask].astype(int),
            CLASSIFIER_LIMIT)
        classifier_columns[threshold] = [union_columns[j] for j in classifier_indices]
        classifier = timeout_classifier(17)
        classifier.fit(union_X[mask][:,classifier_indices],
                       timeout[mask].astype(int))
        classifiers[threshold] = classifier
    selected = set(global_columns)
    selected.update(c for part in specialist_columns.values() for c in part)
    selected.update(c for part in classifier_columns.values() for c in part)
    selected_columns = [c for c in union_columns if c in selected]
    artifact = {
        'artifact_version':artifact_version,
        'columns':selected_columns,
        'global_columns':global_columns,
        'specialist_columns_by_threshold':specialist_columns,
        'classifier_columns_by_threshold':classifier_columns,
        'estimator':global_model,
        'global_estimator':global_model,
        'threshold_specialists':specialists,
        'threshold_specialist_weight':SPECIALIST_WEIGHT,
        'timeout_classifiers':classifiers,
        'timeout_probability_cutoff':TIMEOUT_CUTOFF,
        'reset_family_references':references,
        'template_analogue_bank':template_bank,
        'view':('categorical_pruned_union_geometry_chi_structural_dag_angle_threshold_experts'
                if artifact_version.endswith('_v8') else
                'pruned_union_geometry_chi_structural_dag_angle_threshold_experts'),
        'setting_encoding': ('one_hot_categorical' if artifact_version.endswith('_v8')
                             else 'ordinal_and_log2'),
        'chi_walk_budget_s':3.0,
        'chi_walk_large_cutoff_bytes':40_000_000,
        'chi_walk_rotation_tolerance_rad':BASIS_ROTATION_TOLERANCE,
        'training_rows':int(len(y)),
        'known_thresholds':THRESHOLDS,
        'interval_alpha':.10,
        'interval_log10_radius':float(interval_log10_radius),
    }
    joblib.dump(artifact,ARTIFACT,compress=3)
    return {
        'selected_unique_columns':len(selected_columns),
        'global_columns':len(global_columns),
        'specialist_columns_by_threshold':{str(k):len(v) for k,v in specialist_columns.items()},
        'classifier_columns_by_threshold':{str(k):len(v) for k,v in classifier_columns.items()},
        'selected_columns':selected_columns,
    }


def main():
    rows, parallel_X, parallel_columns, y, timeout, thresholds, names = load_training_table()
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    walks = json.loads((ROOT / 'research' / 'chi_walk_angle_cache.json').read_text())['tables']
    for name,walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    extended_X, extended_columns = load_extended_matrix(rows,thresholds)
    union_X = np.column_stack((parallel_X,extended_X))
    union_columns = parallel_columns+extended_columns
    retained,constant,duplicate = redundant_columns(union_X,union_columns)
    parallel_retained = retained[retained < len(parallel_columns)]
    parallel_X = parallel_X[:,parallel_retained]
    parallel_columns = [parallel_columns[j] for j in parallel_retained]
    union_X = union_X[:,retained]
    union_columns = [union_columns[j] for j in retained]
    fold_by_name = load_fold_table()
    predictions = {}
    report = {
        'artifact_version':'pruned_union_threshold_experts_v7',
        'rows':len(rows),
        'circuits':len(set(names)),
        'parallel_features':len(parallel_columns),
        'extended_features':len(union_columns)-len(parallel_columns),
        'extended_candidate_features':len(extended_columns),
        'union_features':len(union_columns),
        'original_union_features':len(retained)+len(constant)+len(duplicate),
        'constant_columns':constant,
        'exact_duplicate_columns':duplicate,
        'feature_limits':{'global':GLOBAL_LIMIT,'specialist':SPECIALIST_LIMIT,
                          'classifier':CLASSIFIER_LIMIT},
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
            'near_basis_calibration':metrics(y,timeout,thresholds,pred['basis_adjusted_seconds']),
            'basis_vs_template':paired_bootstrap(
                y,timeout,names,fold_by_name,split,pred['template_adjusted_seconds'],
                pred['basis_adjusted_seconds']),
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
                pred['basis_adjusted_seconds']),
        }
        report['splits'][split] = split_report
        print(split,json.dumps(split_report,indent=2),flush=True)

    with OOF.open('w',newline='') as file:
        fields = ['filename','threshold','status','actual_s']
        for split in ('matched','structural'):
            fields += [f'{split}_global_pred_s',f'{split}_specialist_pred_s',
                       f'{split}_timeout_probability',f'{split}_full_union_pred_s',
                       f'{split}_reset_floor_pred_s',f'{split}_both_floors_pred_s',
                       f'{split}_template_pred_s',f'{split}_basis_pred_s']
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
                    f'{split}_basis_pred_s':predictions[split]['basis_adjusted_seconds'][i],
                })
            writer.writerow(entry)
    matched_seconds = predictions['matched']['basis_adjusted_seconds']
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
    report['selected_schema'] = fit_artifact(
        parallel_X,union_X,parallel_columns,union_columns,
        y,timeout,thresholds,interval_radius,
        reset_family_references(rows,features,np.ones(len(rows),dtype=bool)),
        build_template_bank(rows,features,np.ones(len(rows),dtype=bool)))
    report['artifact'] = str(ARTIFACT.relative_to(ROOT))
    report['oof_predictions'] = str(OOF.relative_to(ROOT))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'artifact':str(ARTIFACT),'report':str(REPORT),'oof':str(OOF)},indent=2))


if __name__ == '__main__':
    main()
