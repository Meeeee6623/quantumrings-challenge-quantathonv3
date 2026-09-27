"""Fit the fixed 120-column production schema using uv.

    uv run --locked python research/train_production_model.py

The schema is frozen in quantathon-harness/production_features.json so a
subsequent refit cannot silently select a different feature set. Validation
metrics are printed, not written as a report.
"""
from __future__ import annotations

import csv
import json

import joblib
import numpy as np

from probe_categorical_setting import categorical_matrices
from train_full_union_model import (
    ROOT, build_template_bank, cross_predict, fit_artifact,
    load_extended_matrix, load_fold_table, load_training_table,
    reset_family_references,
)
from train_merged_model import metrics
from chi_walk import model_features

SCHEMA = ROOT / 'quantathon-harness/production_features.json'
ARTIFACT = ROOT / 'quantathon-harness/artifacts/runtime_model.joblib'
VERSION = 'categorical_setting_threshold_experts_v9'
OOF = ROOT / 'research/production_model_oof.csv'


def main():
    schema = json.loads(SCHEMA.read_text())
    selected = schema['columns']
    if len(selected) != 120 or len(set(selected)) != 120:
        raise ValueError('Production schema must contain 120 unique columns')
    if {f'setting_{t}' for t in (16, 64, 512)} - set(selected):
        raise ValueError('Production schema must retain every setting category')
    if {'threshold', 'log_threshold', 'extended__threshold'} & set(selected):
        raise ValueError('Ordinal settings cannot enter the production model')
    rows, primary, p_names, y, timeout, thresholds, names = load_training_table()
    extended, e_names = load_extended_matrix(rows, thresholds)
    p, p_columns, union, union_columns, _, _ = categorical_matrices(
        primary, p_names, extended, e_names, thresholds)
    missing = set(selected) - set(union_columns)
    if missing:
        raise ValueError(f'Schema columns missing from extractor: {sorted(missing)}')
    p_indices = [j for j, name in enumerate(p_columns) if name in selected]
    u_indices = [j for j, name in enumerate(union_columns) if name in selected]
    p, p_columns = p[:, p_indices], [p_columns[j] for j in p_indices]
    union, union_columns = union[:, u_indices], [union_columns[j] for j in u_indices]
    features = json.loads((ROOT / 'research/features.json').read_text())
    walks = json.loads((ROOT / 'research/chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    folds = load_fold_table()
    predictions = {}
    for split in ('matched', 'structural'):
        result = cross_predict(p, union, y, timeout, thresholds, names, rows,
                               features, folds, split)
        predicted = result['basis_adjusted_seconds']
        predictions[split] = predicted
        after = metrics(y, timeout, thresholds, predicted)
        print(f'{split}: score {after["score"]:.6f}; '
              f'timeout {after["timeout_score"]:.6f}; '
              f'tenfold misses {after["tenfold_misses"]}',
              flush=True)
    with OOF.open('w', newline='') as handle:
        writer = csv.DictWriter(handle,
            fieldnames=['filename', 'threshold', 'status', 'actual_s',
                        'matched_pred_s', 'structural_pred_s'],
            lineterminator='\n')
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({'filename': row['filename'],
                             'threshold': row['threshold'],
                             'status': row['status'], 'actual_s': 10**y[i],
                             'matched_pred_s': predictions['matched'][i],
                             'structural_pred_s': predictions['structural'][i]})
    radius = float(np.quantile(np.abs(
        np.log10(np.maximum(1e-9, predictions['matched'])) - y), .90))
    selected_schema = fit_artifact(
        p, union, p_columns, union_columns, y, timeout, thresholds, radius,
        reset_family_references(rows, features,
                                np.ones(len(rows), dtype=bool)),
        build_template_bank(rows, features,
                            np.ones(len(rows), dtype=bool)), VERSION)
    artifact = joblib.load(ARTIFACT)
    if set(artifact['columns']) != set(selected):
        raise AssertionError('Fitted columns differ from fixed production schema')
    print(f'Fitted {selected_schema["selected_unique_columns"]} production inputs; '
          f'interval radius {radius:.6f}; artifact {ARTIFACT}', flush=True)


if __name__ == '__main__':
    main()
