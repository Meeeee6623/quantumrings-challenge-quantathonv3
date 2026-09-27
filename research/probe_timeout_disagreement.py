"""Stress-test a guarded timeout/regression compromise on alternate grouped folds.

The fixed structural split suggested false timeout caps on large threshold-64
circuits.  These alternate circuit-grouped folds test whether the proposed
postprocessing survives different training/holdout assignments.  No artifact
is overwritten.

    uv run --locked python research/probe_timeout_disagreement.py
"""
from __future__ import annotations

import json

import numpy as np

from train_full_union_model import (
    CAP, ROOT, SPECIALIST_WEIGHT, TIMEOUT_CUTOFF, cross_predict,
    floor_fold_predictions, large_work_predictions, load_extended_matrix,
    load_training_table, near_basis_predictions, template_fold_predictions,
)
from train_merged_model import metrics
from chi_walk import model_features

REPORT = ROOT / 'research' / 'timeout_disagreement_probe.json'
SEEDS = (81, 99, 113)


def alternate_folds(names, seed):
    unique = sorted(set(names))
    permuted = np.random.default_rng(seed).permutation(unique)
    return {str(name): {'matched_fold': int(i % 5),
                        'structural_stress_fold': int(i % 5)}
            for i, name in enumerate(permuted)}


def apply_candidate(pred, rows, features, fold_by_name, names, thresholds,
                    bytes_by_row, alpha):
    raw = np.power(10.0, np.clip(
        (1-SPECIALIST_WEIGHT)*pred['global_log']
        + SPECIALIST_WEIGHT*pred['specialist_log'], -9, 9))
    probability = pred['timeout_probability']
    routed = probability >= TIMEOUT_CUTOFF
    seconds = np.where(routed, CAP, raw)
    changed = (routed & (thresholds == 64) & (bytes_by_row > 1_000_000)
               & (raw < 100))
    seconds[changed] = raw[changed]**(1-alpha) * CAP**alpha
    assigned = np.asarray([fold_by_name[str(name)]['matched_fold'] for name in names])
    seconds = floor_fold_predictions(seconds, rows, features, assigned)
    seconds = large_work_predictions(seconds, rows, features)
    seconds = template_fold_predictions(seconds, rows, features, assigned)
    seconds = near_basis_predictions(seconds, rows, features)
    return seconds, changed


def main():
    rows, parallel_X, parallel_columns, y, timeout, thresholds, names = load_training_table()
    extended_X, extended_columns = load_extended_matrix(rows, thresholds)
    union_X = np.column_stack((parallel_X, extended_X))
    # Match selected v6's label-free constant/duplicate filtering.
    from feature_pruning import redundant_columns
    retained, _, _ = redundant_columns(union_X, parallel_columns+extended_columns)
    parallel_X = parallel_X[:, retained[retained < len(parallel_columns)]]
    union_X = union_X[:, retained]
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    walks = json.loads((ROOT / 'research' / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    bytes_by_row = np.asarray([features[row['filename']]['qasm_bytes'] for row in rows])
    report = {'method': 'alternate five-fold circuit-grouped assignments',
              'seeds': {}, 'candidate': {
                  'threshold': 64, 'min_qasm_bytes': 1_000_000,
                  'max_uncapped_regression_seconds': 100,
                  'soft_cap_log_weight': .2}}
    for seed in SEEDS:
        folds = alternate_folds(names, seed)
        pred = cross_predict(parallel_X, union_X, y, timeout, thresholds,
                             names, rows, features, folds, 'matched')
        result = {'baseline': metrics(y, timeout, thresholds,
                                      pred['basis_adjusted_seconds'])}
        for label, alpha in [('soft', .2), ('uncap', 0.0)]:
            seconds, changed = apply_candidate(pred, rows, features, folds,
                                                names, thresholds, bytes_by_row,
                                                alpha)
            result[label] = metrics(y, timeout, thresholds, seconds)
            result[label]['changed_rows'] = int(changed.sum())
            result[label]['changed_true_timeouts'] = int(np.sum(changed & timeout))
        report['seeds'][str(seed)] = result
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(seed, *(f'{label}={result[label]["score"]:.6f}'
                      for label in ('baseline','soft','uncap')),
              f'changed={result["soft"]["changed_rows"]}',
              f'true_timeouts={result["soft"]["changed_true_timeouts"]}',
              flush=True)


if __name__ == '__main__':
    main()
