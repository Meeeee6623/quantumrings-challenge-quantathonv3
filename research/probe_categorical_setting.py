"""Compare ordinal and one-hot simulator settings without changing v7.

    uv run --locked python research/probe_categorical_setting.py

The setting is categorical in the one-hot arm. Threshold specialists and
timeout classifiers still use the discrete setting to select their model.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from train_full_union_model import (
    ROOT, cross_predict, load_extended_matrix, load_fold_table,
    load_training_table,
)
from train_merged_model import metrics, paired_bootstrap
from chi_walk import model_features
from feature_pruning import redundant_columns

OUT = ROOT / 'research'
REPORT = OUT / 'categorical_setting_probe.json'
OOF = OUT / 'categorical_setting_oof.csv'
SETTINGS = (16, 64, 512)


def categorical_matrices(primary, p_names, extended, e_names, thresholds):
    """Replace every numeric setting field, including the secondary duplicate."""
    p_keep = [i for i, name in enumerate(p_names)
              if name not in ('threshold', 'log_threshold')]
    e_keep = [i for i, name in enumerate(e_names)
              if name != 'extended__threshold']
    one_hot = np.column_stack([thresholds == setting for setting in SETTINGS]).astype(float)
    p = np.column_stack((primary[:, p_keep], one_hot))
    p_columns = [p_names[i] for i in p_keep] + [f'setting_{t}' for t in SETTINGS]
    union = np.column_stack((p, extended[:, e_keep]))
    union_columns = p_columns + [e_names[i] for i in e_keep]
    retained, constants, duplicate = redundant_columns(union, union_columns)
    p_retained = retained[retained < p.shape[1]]
    assert not any(name in union_columns for name in
                   ('threshold', 'log_threshold', 'extended__threshold'))
    assert all(name in union_columns for name in
               ('setting_16', 'setting_64', 'setting_512'))
    return (p[:, p_retained], [p_columns[i] for i in p_retained],
            union[:, retained], [union_columns[i] for i in retained],
            constants, duplicate)


def main():
    rows, primary, p_names, y, timeout, thresholds, names = load_training_table()
    extended, e_names = load_extended_matrix(rows, thresholds)
    p, p_columns, union, union_columns, constants, duplicate = categorical_matrices(
        primary, p_names, extended, e_names, thresholds)
    features = json.loads((OUT / 'features.json').read_text())
    walks = json.loads((OUT / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    folds = load_fold_table()
    baseline = list(csv.DictReader((OUT / 'full_union_model_oof.csv').open()))
    assert [(row['filename'], int(row['threshold'])) for row in rows] == [
        (row['filename'], int(row['threshold'])) for row in baseline]
    report = {'encoding': 'one-hot (16, 64, 512); no ordinal or log threshold input',
              'rows': len(rows), 'primary_columns': len(p_columns),
              'union_columns': len(union_columns), 'constant_columns': constants,
              'exact_duplicate_columns': duplicate, 'splits': {}}
    saved = {}
    for split in ('matched', 'structural'):
        predictions = cross_predict(p, union, y, timeout, thresholds, names,
                                    rows, features, folds, split)
        current = np.asarray([float(row[f'{split}_basis_pred_s']) for row in baseline])
        categorical = predictions['basis_adjusted_seconds']
        report['splits'][split] = {
            'current_ordinal_v7': metrics(y, timeout, thresholds, current),
            'categorical': metrics(y, timeout, thresholds, categorical),
            'paired': paired_bootstrap(y, timeout, names, folds, split,
                                       current, categorical),
        }
        saved[split] = categorical
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(split, report['splits'][split]['current_ordinal_v7']['score'],
              report['splits'][split]['categorical']['score'], flush=True)
    with OOF.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['filename','threshold','status',
                                                    'actual_s','matched_pred_s',
                                                    'structural_pred_s'],
                                lineterminator='\n')
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({'filename': row['filename'],
                             'threshold': row['threshold'],
                             'status': row['status'],
                             'actual_s': float(10**y[i]),
                             'matched_pred_s': saved['matched'][i],
                             'structural_pred_s': saved['structural'][i]})
    print('Wrote', REPORT, OOF)


if __name__ == '__main__':
    main()
