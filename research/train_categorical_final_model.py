"""Fit the categorical simulator-setting release model after grouped probes.

    uv run --locked python research/train_categorical_final_model.py
    uv run --locked python research/train_categorical_final_model.py --pruned
"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np

from probe_categorical_setting import OUT, categorical_matrices
from train_full_union_model import (
    build_template_bank, cross_predict, fit_artifact, load_extended_matrix,
    load_fold_table, load_training_table, reset_family_references,
)
from train_merged_model import metrics, paired_bootstrap
from chi_walk import model_features

REPORT = OUT / 'categorical_model_validation.json'
OOF = OUT / 'categorical_model_oof.csv'
VERSION = 'categorical_setting_threshold_experts_v8'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned', action='store_true',
                        help='fit only audit-proposed keep features')
    args = parser.parse_args()
    rows, primary, p_names, y, timeout, thresholds, names = load_training_table()
    extended, e_names = load_extended_matrix(rows, thresholds)
    p, p_columns, union, union_columns, constants, duplicates = categorical_matrices(
        primary, p_names, extended, e_names, thresholds)
    if args.pruned:
        audit = list(csv.DictReader((OUT / 'feature_audit/feature_decisions.csv').open()))
        allowed = {row['feature'] for row in audit if row['decision'] == 'keep'}
        allowed.update(f'setting_{setting}' for setting in (16, 64, 512))
        pi = [j for j, name in enumerate(p_columns) if name in allowed]
        ui = [j for j, name in enumerate(union_columns) if name in allowed]
        p = p[:, pi]
        p_columns = [p_columns[j] for j in pi]
        union = union[:, ui]
        union_columns = [union_columns[j] for j in ui]
    features = json.loads((OUT / 'features.json').read_text())
    walks = json.loads((OUT / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    folds = load_fold_table()
    old = list(csv.DictReader((OUT / 'full_union_model_oof.csv').open()))
    assert [(row['filename'], int(row['threshold'])) for row in rows] == [
        (row['filename'], int(row['threshold'])) for row in old]
    report = {'artifact_version': VERSION,
              'setting_encoding': 'one-hot categorical (16, 64, 512)',
              'audit_pruned': args.pruned, 'rows': len(rows),
              'circuits': len(set(names)), 'primary_candidates': len(p_columns),
              'union_candidates': len(union_columns),
              'constant_columns': constants, 'exact_duplicate_columns': duplicates,
              'splits': {}}
    saved = {}
    for split in ('matched', 'structural'):
        pred = cross_predict(p, union, y, timeout, thresholds, names, rows,
                             features, folds, split)
        saved[split] = pred['basis_adjusted_seconds']
        prior = np.asarray([float(row[f'{split}_basis_pred_s']) for row in old])
        report['splits'][split] = {
            'current_v7': metrics(y, timeout, thresholds, prior),
            'categorical_v8': metrics(y, timeout, thresholds, saved[split]),
            'paired': paired_bootstrap(y, timeout, names, folds, split,
                                       prior, saved[split]),
        }
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(split, report['splits'][split]['categorical_v8']['score'], flush=True)
    with OOF.open('w', newline='') as handle:
        writer = csv.DictWriter(handle,
            fieldnames=['filename','threshold','status','actual_s',
                        'matched_pred_s','structural_pred_s'],
            lineterminator='\n')
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({'filename': row['filename'],
                             'threshold': row['threshold'],
                             'status': row['status'], 'actual_s': 10**y[i],
                             'matched_pred_s': saved['matched'][i],
                             'structural_pred_s': saved['structural'][i]})
    radius = float(np.quantile(np.abs(
        np.log10(np.maximum(1e-9, saved['matched'])) - y), .90))
    report['uncertainty'] = {'interval_alpha': .10,
                             'interval_log10_radius': radius,
                             'multiplicative_factor': 10**radius}
    report['selected_schema'] = fit_artifact(
        p, union, p_columns, union_columns, y, timeout, thresholds, radius,
        reset_family_references(rows, features,
                                np.ones(len(rows), dtype=bool)),
        build_template_bank(rows, features,
                            np.ones(len(rows), dtype=bool)), VERSION)
    report['artifact'] = 'quantathon-harness/artifacts/runtime_model.joblib'
    report['oof_predictions'] = str(OOF.relative_to(OUT.parent))
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'selected_features':
                      report['selected_schema']['selected_unique_columns'],
                      'report': str(REPORT), 'oof': str(OOF)}, indent=2))


if __name__ == '__main__':
    main()
