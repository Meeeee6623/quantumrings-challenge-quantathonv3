"""Nested, circuit-grouped feature-pruning experiment for the union model.

Feature ranks are learned on each training fold only.  Constant and exactly
duplicate columns are removed without consulting runtime labels.  This probe
does not overwrite the deployed artifact or its validation files.

    uv run --locked python research/probe_feature_pruning.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np

from train_full_union_model import (
    CAP, ROOT, SPECIALIST_WEIGHT, THRESHOLDS, TIMEOUT_CUTOFF,
    floor_fold_predictions, large_work_predictions,
    load_extended_matrix, load_fold_table, load_training_table,
    near_basis_predictions, template_fold_predictions,
)
from train_merged_model import (
    global_regressor, metrics, positive_probability, threshold_regressor,
    timeout_classifier,
)
from chi_walk import model_features
from feature_pruning import importance_choice, redundant_columns

REPORT = ROOT / 'research' / 'feature_pruning_validation.json'
CANDIDATES = {
    'deduplicated': (None, None, None),
    'moderate': (120, 200, 80),
    'aggressive': (80, 140, 60),
}


def predict_candidate(P, U, y, timeout, thresholds, names, rows, features,
                      fold_by_name, split, limits):
    assigned = np.asarray([int(fold_by_name[name][
        'matched_fold' if split == 'matched' else 'structural_stress_fold'])
        for name in names])
    global_log = np.zeros(len(y))
    specialist_log = np.zeros(len(y))
    probability = np.zeros(len(y))
    selected_counts = defaultdict(list)
    for fold in sorted(set(assigned)):
        train, test = assigned != fold, assigned == fold
        g_cols = importance_choice(global_regressor(17), P[train], y[train], limits[0])
        selected_counts['global'].append(len(g_cols))
        global_fit = global_regressor(17).fit(P[train][:, g_cols], y[train])
        global_log[test] = global_fit.predict(P[test][:, g_cols])
        for threshold in THRESHOLDS:
            tr = train & (thresholds == threshold)
            te = test & (thresholds == threshold)
            if not te.any():
                continue
            s_cols = importance_choice(threshold_regressor(17+fold), U[tr], y[tr], limits[1])
            selected_counts['specialist'].append(len(s_cols))
            specialist = threshold_regressor(17+fold).fit(U[tr][:, s_cols], y[tr])
            specialist_log[te] = specialist.predict(U[te][:, s_cols])
            c_cols = importance_choice(timeout_classifier(17+fold), U[tr],
                                       timeout[tr].astype(int), limits[2])
            selected_counts['classifier'].append(len(c_cols))
            classifier = timeout_classifier(17+fold).fit(
                U[tr][:, c_cols], timeout[tr].astype(int))
            probability[te] = positive_probability(classifier, U[te][:, c_cols])
    seconds = np.power(10.0, np.clip((1-SPECIALIST_WEIGHT)*global_log
                                      + SPECIALIST_WEIGHT*specialist_log, -9, 9))
    seconds = np.where(probability >= TIMEOUT_CUTOFF, CAP, seconds)
    adjusted = floor_fold_predictions(seconds, rows, features, assigned)
    adjusted = large_work_predictions(adjusted, rows, features)
    adjusted = template_fold_predictions(adjusted, rows, features, assigned)
    adjusted = near_basis_predictions(adjusted, rows, features)
    return adjusted, dict(selected_counts)


def main():
    rows, P_full, p_columns, y, timeout, thresholds, names = load_training_table()
    E, e_columns = load_extended_matrix(rows, thresholds)
    columns = p_columns + e_columns
    U_full = np.column_stack((P_full, E))
    keep, constant, duplicate = redundant_columns(U_full, columns)
    p_keep = keep[keep < P_full.shape[1]]
    P, U = P_full[:, p_keep], U_full[:, keep]
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    walks = json.loads((ROOT / 'research' / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    fold_by_name = load_fold_table()
    report = {
        'method': 'Training-fold-only impurity-rank selection; fixed circuit-grouped folds',
        'original_columns': len(columns),
        'deduplicated_columns': len(keep),
        'original_parallel_columns': len(p_columns),
        'deduplicated_parallel_columns': len(p_keep),
        'constant_columns': constant,
        'exact_duplicate_columns': duplicate,
        'candidates': {},
    }
    for split in ('matched', 'structural'):
        seconds, _ = predict_candidate(P_full, U_full, y, timeout,
                                       thresholds, names, rows, features,
                                       fold_by_name, split, (None, None, None))
        report.setdefault('baseline', {})[split] = metrics(y, timeout, thresholds, seconds)
    for label, limits in CANDIDATES.items():
        start = time.monotonic()
        result = {'limits': {'global': limits[0], 'specialist': limits[1],
                             'classifier': limits[2]}, 'splits': {}}
        for split in ('matched', 'structural'):
            pred, counts = predict_candidate(P, U, y, timeout, thresholds, names,
                                              rows, features, fold_by_name, split, limits)
            result['splits'][split] = metrics(y, timeout, thresholds, pred)
            result['splits'][split]['selected_column_counts'] = {
                k: sorted(set(v)) for k, v in counts.items()}
            print(label, split, result['splits'][split]['score'], flush=True)
        result['elapsed_seconds'] = round(time.monotonic()-start, 1)
        report['candidates'][label] = result
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('report', REPORT)


if __name__ == '__main__':
    main()
