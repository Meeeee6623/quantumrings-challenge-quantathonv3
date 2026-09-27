"""Grouped ablation of audit-proposed feature drops under categorical settings.

    uv run --locked python research/probe_feature_audit_prune.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from probe_categorical_setting import OUT, categorical_matrices
from train_full_union_model import (
    cross_predict, load_extended_matrix, load_fold_table, load_training_table)
from train_merged_model import metrics, paired_bootstrap
from chi_walk import model_features

REPORT = OUT / 'feature_audit_prune_probe.json'


def main():
    rows, primary, p_names, y, timeout, thresholds, names = load_training_table()
    extended, e_names = load_extended_matrix(rows, thresholds)
    p, p_columns, union, union_columns, _, _ = categorical_matrices(
        primary, p_names, extended, e_names, thresholds)
    audit = list(csv.DictReader((OUT / 'feature_audit/feature_decisions.csv').open()))
    allowed = {row['feature'] for row in audit if row['decision'] == 'keep'}
    allowed.update({f'setting_{setting}' for setting in (16, 64, 512)})
    p_index = [j for j, name in enumerate(p_columns) if name in allowed]
    union_index = [j for j, name in enumerate(union_columns) if name in allowed]
    pruned_p, pruned_union = p[:, p_index], union[:, union_index]
    features = json.loads((OUT / 'features.json').read_text())
    walks = json.loads((OUT / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    folds = load_fold_table()
    categorical = list(csv.DictReader((OUT / 'categorical_setting_oof.csv').open()))
    assert [(row['filename'], int(row['threshold'])) for row in rows] == [
        (row['filename'], int(row['threshold'])) for row in categorical]
    report = {'source_feature_count': len(audit),
              'pruned_primary_count': len(p_index),
              'pruned_union_count': len(union_index),
              'dropped_from_source': [row['feature'] for row in audit
                                      if row['decision'] == 'drop'],
              'numeric_setting_fields_removed_by_categorical_encoding':
                  ['threshold', 'log_threshold'],
              'splits': {}}
    for split in ('matched', 'structural'):
        pred = cross_predict(pruned_p, pruned_union, y, timeout, thresholds,
                             names, rows, features, folds, split)
        baseline = np.asarray([float(row[f'{split}_pred_s']) for row in categorical])
        pruned = pred['basis_adjusted_seconds']
        report['splits'][split] = {
            'categorical_full': metrics(y, timeout, thresholds, baseline),
            'categorical_audit_pruned': metrics(y, timeout, thresholds, pruned),
            'paired': paired_bootstrap(y, timeout, names, folds, split,
                                       baseline, pruned)}
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(split, report['splits'][split]['categorical_full']['score'],
              report['splits'][split]['categorical_audit_pruned']['score'],
              flush=True)


if __name__ == '__main__':
    main()
