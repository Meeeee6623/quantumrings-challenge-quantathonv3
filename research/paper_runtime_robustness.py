"""Repeat the paper runtime comparison under new neural seeds.

    uv run --locked --extra paper python research/paper_runtime_robustness.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from paper_runtime_replication import (
    OUT, evaluate, load_paper_table, train_fold,
)
from train_full_union_model import load_fold_table
from train_merged_model import paired_bootstrap

REPORT = OUT / 'paper_runtime_robustness.json'
OOF = OUT / 'paper_runtime_oof.csv'
SEEDS = (2700, 3700)


def fit_split(X, y, family, names, assigned, mode, columns, seed_base):
    output = np.zeros(len(y))
    for fold in sorted(set(assigned)):
        train = np.flatnonzero(assigned != fold)
        test = np.flatnonzero(assigned == fold)
        result, _ = train_fold(X[:, columns], y, family, names, train, test,
                               mode, seed_base + int(fold))
        output[test] = result
        print('fold', fold, 'seed', seed_base, 'mode', mode,
              'columns', len(columns), flush=True)
    return np.power(10, np.clip(output, -8, 8))


def main():
    rows, X, y, timeout, thresholds, names, family, _ = load_paper_table()
    existing = list(csv.DictReader(OOF.open()))
    assert [(r['filename'], int(r['threshold'])) for r in rows] == [
        (r['filename'], int(r['threshold'])) for r in existing]
    folds = load_fold_table()
    report = {'method': 'repeat circuit-grouped neural fits with independent seeds; same feature and family inputs',
              'seeds': {}, 'threshold_omission': {}, 'unknown_family_guard': {}}
    guessed = {row['filename']: int(row['outside_reference_95pct'])
               for row in csv.DictReader((OUT / 'algorithm_geometry_guesses.csv').open())}
    unknown_family = np.asarray([20 if guessed[str(name)] else family[i]
                                 for i, name in enumerate(names)], dtype=np.int64)
    for split in ('matched', 'structural'):
        key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
        assigned = np.asarray([int(folds[str(name)][key]) for name in names])
        baseline_graph = np.asarray([float(row[f'{split}_mlp_graph_pred_s'])
                                     for row in existing])
        baseline_family = np.asarray([float(row[f'{split}_family_film_residual_pred_s'])
                                      for row in existing])
        report['seeds'].setdefault(split, {})['1700'] = {
            'graph': evaluate(y, baseline_graph, timeout, thresholds),
            'family': evaluate(y, baseline_family, timeout, thresholds),
        }
        for seed in SEEDS:
            graph = fit_split(X, y, family, names, assigned, 'plain',
                              list(range(33)), seed)
            family_pred = fit_split(X, y, family, names, assigned, 'family',
                                    list(range(33)), seed)
            report['seeds'][split][str(seed)] = {
                'graph': evaluate(y, graph, timeout, thresholds),
                'family': evaluate(y, family_pred, timeout, thresholds),
                'paired_bootstrap': paired_bootstrap(
                    y, timeout, names, folds, split, graph, family_pred),
            }
            REPORT.write_text(json.dumps(report, indent=2) + '\n')
            print(split, seed, 'graph', report['seeds'][split][str(seed)]['graph']['score'],
                  'family', report['seeds'][split][str(seed)]['family']['score'],
                  flush=True)
        guarded = fit_split(X, y, unknown_family, names, assigned, 'family',
                            list(range(33)), 1700)
        report['unknown_family_guard'][split] = {
            'unknown_circuits': int(sum(guessed.values())),
            'score': evaluate(y, guarded, timeout, thresholds),
            'paired_vs_forced_family': paired_bootstrap(
                y, timeout, names, folds, split, baseline_family, guarded),
        }
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(split, 'unknown-family guard score',
              report['unknown_family_guard'][split]['score']['score'], flush=True)
    # Paper Section IV-E says runtime changes little when threshold is omitted.
    # Test the same family architecture with this input removed on matched folds.
    assigned = np.asarray([int(folds[str(name)]['matched_fold']) for name in names])
    no_threshold = fit_split(X, y, family, names, assigned, 'family',
                             list(range(32)), 1700)
    with_threshold = np.asarray([float(row['matched_family_film_residual_pred_s'])
                                 for row in existing])
    report['threshold_omission'] = {
        'with_threshold': evaluate(y, with_threshold, timeout, thresholds),
        'without_threshold': evaluate(y, no_threshold, timeout, thresholds),
        'paired_bootstrap': paired_bootstrap(
            y, timeout, names, folds, 'matched', with_threshold, no_threshold),
    }
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('threshold ablation', report['threshold_omission']['with_threshold']['score'],
          report['threshold_omission']['without_threshold']['score'], flush=True)


if __name__ == '__main__':
    main()
