"""Test stronger same-signature analogue weighting on grouped folds.

    uv run --locked python research/probe_template_weight.py
"""
from __future__ import annotations

import csv
import json
import statistics

import numpy as np

from feature_pruning import redundant_columns
from probe_timeout_disagreement import alternate_folds
from train_full_union_model import (
    ROOT, build_template_bank, cross_predict, load_extended_matrix,
    load_fold_table, load_training_table, near_basis_predictions,
)
from train_merged_model import metrics, paired_bootstrap
from template_analogues import signature
from chi_walk import model_features

REPORT = ROOT / 'research' / 'template_weight_probe.json'
SEEDS = (137, 149, 163)
WEIGHTS = (.5, .75, 1.0)


def analogue_values(rows, features, folds, names, split):
    key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
    assigned = np.asarray([int(folds[str(name)][key]) for name in names])
    analogues = np.full(len(rows), np.nan)
    counts = np.zeros(len(rows), dtype=int)
    for fold in sorted(set(assigned)):
        bank = build_template_bank(rows, features, assigned != fold)
        for i in np.flatnonzero(assigned == fold):
            row = rows[i]
            values = bank.get((int(row['threshold']),
                               *signature(features[row['filename']])), ())
            if len(values) >= 2:
                analogues[i] = 10**statistics.median(values)
                counts[i] = len(values)
    return analogues, counts


def evaluate(base, analogues, counts, rows, features, y, timeout, thresholds):
    eligible = np.isfinite(analogues)
    scores = {'eligible_rows': int(eligible.sum()), 'weights': {}}
    predictions = {}
    for weight in WEIGHTS:
        seconds = base.copy()
        seconds[eligible] = (base[eligible]**(1-weight)
                             * analogues[eligible]**weight)
        seconds = near_basis_predictions(seconds, rows, features)
        predictions[weight] = seconds
        scores['weights'][str(weight)] = metrics(y, timeout, thresholds, seconds)
    adaptive = base.copy()
    adaptive[counts == 2] = np.sqrt(base[counts == 2] * analogues[counts == 2])
    adaptive[counts >= 3] = analogues[counts >= 3]
    adaptive = near_basis_predictions(adaptive, rows, features)
    predictions['adaptive'] = adaptive
    scores['weights']['adaptive'] = metrics(y, timeout, thresholds, adaptive)
    scores['eligible_by_reference_count'] = {
        str(count): int(np.sum(counts == count)) for count in sorted(set(counts))
        if count >= 2}
    return scores, predictions


def main():
    rows, parallel_X, parallel_columns, y, timeout, thresholds, names = load_training_table()
    extended_X, extended_columns = load_extended_matrix(rows, thresholds)
    union_X = np.column_stack((parallel_X, extended_X))
    retained, _, _ = redundant_columns(union_X, parallel_columns + extended_columns)
    parallel_X = parallel_X[:, retained[retained < len(parallel_columns)]]
    union_X = union_X[:, retained]
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    walks = json.loads((ROOT / 'research' / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    report = {'method': 'five-fold circuit-grouped OOF; training-fold-only analogue bank',
              'fixed': {}, 'alternate_seeds': {}}
    folds = load_fold_table()
    oof = list(csv.DictReader((ROOT / 'research' / 'full_union_model_oof.csv').open()))
    assert [(row['filename'], int(row['threshold'])) for row in rows] == [
        (row['filename'], int(row['threshold'])) for row in oof]
    for split in ('matched', 'structural'):
        analogues, counts = analogue_values(rows, features, folds, names, split)
        base = np.asarray([float(row[f'{split}_both_floors_pred_s']) for row in oof])
        scores, predictions = evaluate(base, analogues, counts, rows, features,
                                       y, timeout, thresholds)
        saved = np.asarray([float(row[f'{split}_basis_pred_s']) for row in oof])
        assert np.allclose(predictions['adaptive'], saved, rtol=1e-12, atol=1e-12)
        scores['paired_bootstrap_direct_vs_half'] = paired_bootstrap(
            y, timeout, names, folds, split, predictions[.5], predictions[1.0])
        scores['paired_bootstrap_adaptive_vs_half'] = paired_bootstrap(
            y, timeout, names, folds, split, predictions[.5], predictions['adaptive'])
        report['fixed'][split] = scores
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('fixed',*(f'{weight}={report["fixed"]["matched"]["weights"][str(weight)]["score"]:.6f}'
                    for weight in WEIGHTS),flush=True)
    for seed in SEEDS:
        folds = alternate_folds(names, seed)
        prediction = cross_predict(parallel_X, union_X, y, timeout,
                                   thresholds, names, rows, features, folds, 'matched')
        analogues, counts = analogue_values(rows, features, folds, names, 'matched')
        scores, predictions = evaluate(prediction['large_adjusted_seconds'],
                                       analogues, counts, rows, features, y, timeout,
                                       thresholds)
        assert np.allclose(predictions['adaptive'], prediction['basis_adjusted_seconds'],
                           rtol=1e-12, atol=1e-12)
        report['alternate_seeds'][str(seed)] = scores
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(seed, *(f'{weight}={scores["weights"][str(weight)]["score"]:.6f}'
                      for weight in WEIGHTS),
              f'adaptive={scores["weights"]["adaptive"]["score"]:.6f}',
              f'eligible={scores["eligible_rows"]}',flush=True)


if __name__ == '__main__':
    main()
