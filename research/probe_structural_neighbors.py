"""Screen label-free structural nearest-neighbor analogues on saved OOF folds.

This is a research-only postprocessing probe. Neighbor labels are drawn solely
from the training circuits of each circuit-grouped fold.

    uv run --locked python research/probe_structural_neighbors.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from train_full_union_model import (
    ROOT, cross_predict, load_extended_matrix, load_fold_table, load_training_table,
)
from chi_walk import model_features
from feature_pruning import redundant_columns
from probe_timeout_disagreement import alternate_folds
from template_analogues import signature


CORE = ('n_qubits', 'ops', 'two_q', 'multi_q', 'effective_ops', 'depth',
        'resets', 'chi_upper_mid', 'qasm_bytes')
GATES = CORE + ('gate_h', 'gate_x', 'gate_rx', 'gate_ry', 'gate_rz', 'gate_u',
                'gate_u3', 'gate_cx', 'gate_cz', 'gate_cp', 'gate_swap',
                'gate_rzz', 'gate_ccx', 'measurements')
SHAPE = CORE + ('unique_pairs', 'pair_reuse', 'graph_cutwidth_rcm',
                'graph_degree_entropy', 'randomness_proxy',
                'chi_est_log2_peak', 'chi_capacity_fraction',
                'two_q_ratio', 'nonclifford_ratio', 'liveness_peak_window')
SETS = {'core': CORE, 'gates': GATES, 'shape': SHAPE}
LOG_COLUMNS = {key for key in CORE + GATES + SHAPE
               if key not in ('randomness_proxy', 'chi_capacity_fraction',
                              'two_q_ratio', 'nonclifford_ratio',
                              'graph_degree_entropy')}
REPORT = ROOT / 'research' / 'structural_neighbor_probe.json'
CANDIDATES = {
    'gates': {'k': 3, 'radius': 0.8, 'spread_limit': 0.6, 'weight': 0.25},
    'shape': {'k': 3, 'radius': 1.6, 'spread_limit': 0.6, 'weight': 0.25},
}
ALTERNATE_SEEDS = (181, 197, 211)


def feature_matrix(features, names, columns):
    matrix = np.asarray([[float(features[name].get(column, 0))
                          for column in columns] for name in names])
    for j, column in enumerate(columns):
        if column in LOG_COLUMNS:
            matrix[:, j] = np.log1p(np.maximum(matrix[:, j], 0))
    center = np.median(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 0.1] = 0.1
    return (matrix - center) / scale


def challenge_score(actual, predicted):
    loss = np.minimum(1.0, np.abs(np.log10(predicted / actual)) / 2)
    return float(1 - np.mean(loss))


def apply_candidate(baseline, nearest, distances, exact_counts, log_actual,
                    config):
    k = config['k']
    neighbor_values = log_actual[nearest[:, :k]]
    analogue = np.median(neighbor_values, axis=1)
    spread = np.ptp(neighbor_values, axis=1)
    eligible = ((exact_counts < 2) &
                (distances[:, 0] <= config['radius']) &
                (spread <= config['spread_limit']))
    proposed = np.asarray(baseline, dtype=float).copy()
    weight = config['weight']
    proposed[eligible] = np.power(10,
        (1 - weight) * np.log10(proposed[eligible]) +
        weight * analogue[eligible])
    return proposed, int(eligible.sum())


def neighbour_predictions(distance, circuit_names, rows, folds, split,
                          signature_by_name):
    fold_key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
    assigned = np.asarray([int(folds[r['filename']][fold_key]) for r in rows])
    thresholds = np.asarray([int(r['threshold']) for r in rows])
    index = {name: i for i, name in enumerate(circuit_names)}
    circuits = np.asarray([index[r['filename']] for r in rows])
    log_actual = np.log10([float(r['actual_s']) for r in rows])
    nearest = np.full((len(rows), 5), -1, dtype=int)
    distances = np.full((len(rows), 5), math.inf)
    exact_counts = np.zeros(len(rows), dtype=int)
    for fold in sorted(set(assigned)):
        train = assigned != fold
        for threshold in sorted(set(thresholds)):
            train_ix = np.flatnonzero(train & (thresholds == threshold))
            test_ix = np.flatnonzero((assigned == fold) & (thresholds == threshold))
            if not len(train_ix) or not len(test_ix):
                continue
            train_signatures = [signature_by_name[rows[i]['filename']]
                                for i in train_ix]
            for i in test_ix:
                exact_counts[i] = train_signatures.count(
                    signature_by_name[rows[i]['filename']])
            sub = distance[np.ix_(circuits[test_ix], circuits[train_ix])]
            order = np.argsort(sub, axis=1)[:, :5]
            nearest[test_ix] = train_ix[order]
            distances[test_ix] = np.take_along_axis(sub, order, axis=1)
    return nearest, distances, exact_counts, log_actual


def evaluate(rows, nearest, distances, exact_counts, baseline, log_actual):
    actual = np.power(10, log_actual)
    outcomes = []
    for k in (1, 3, 5):
        neighbor_values = log_actual[nearest[:, :k]]
        analogue = np.median(neighbor_values, axis=1)
        spread = np.ptp(neighbor_values, axis=1)
        for radius in (0.4, 0.8, 1.6, math.inf):
            for spread_limit in (0.3, 0.6, math.inf):
                eligible = ((exact_counts < 2) &
                            (distances[:, 0] <= radius) &
                            (spread <= spread_limit))
                for weight in (0.25, 0.5, 0.75):
                    proposed = np.asarray(baseline, dtype=float).copy()
                    proposed[eligible] = np.power(10,
                        (1 - weight) * np.log10(proposed[eligible]) +
                        weight * analogue[eligible])
                    outcomes.append({
                        'k': k, 'radius': radius, 'spread_limit': spread_limit,
                        'weight': weight, 'eligible_rows': int(eligible.sum()),
                        'score': challenge_score(actual, proposed),
                    })
    return outcomes


def main():
    rows = list(csv.DictReader((ROOT / 'research' / 'full_union_model_oof.csv').open()))
    names = sorted(set(row['filename'] for row in rows))
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    folds = load_fold_table()
    signatures = {name: signature(features[name]) for name in names}
    report = {'method': 'training-fold-only same-threshold neighbor labels; label-free feature scaling on all released QASM; exact-template rows untouched',
              'splits': {}, 'selected_configs': CANDIDATES,
              'alternate_seeds': {}}
    distances_by_set = {}
    for set_name, columns in SETS.items():
        matrix = feature_matrix(features, names, columns)
        distance = cdist(matrix, matrix, metric='euclidean') / math.sqrt(len(columns))
        distances_by_set[set_name] = distance
        report['splits'][set_name] = {}
        for split in ('matched', 'structural'):
            nearest, distances, counts, actual = neighbour_predictions(
                distance, names, rows, folds, split, signatures)
            base = np.asarray([float(row[f'{split}_basis_pred_s']) for row in rows])
            results = evaluate(rows, nearest, distances, counts, base, actual)
            results.sort(key=lambda item: item['score'], reverse=True)
            baseline_score = challenge_score(np.power(10, actual), base)
            report['splits'][set_name][split] = {
                'baseline_score': baseline_score,
                'top_10': results[:10],
                'all': results,
                'rows_without_exact_template': int(np.sum(counts < 2)),
            }
            best = results[0]
            print(set_name, split, f'baseline={baseline_score:.6f}',
                  f'best={best["score"]:.6f}', f'changed={best["eligible_rows"]}',
                  f'config={best}', flush=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')

    # New grouped assignments test only the two candidates selected above.
    training, parallel_X, parallel_columns, y, timeout, thresholds, row_names = load_training_table()
    assert [(r['filename'], int(r['threshold'])) for r in training] == [
        (r['filename'], int(r['threshold'])) for r in rows]
    extended_X, extended_columns = load_extended_matrix(training, thresholds)
    union_X = np.column_stack((parallel_X, extended_X))
    retained, _, _ = redundant_columns(union_X, parallel_columns + extended_columns)
    parallel_X = parallel_X[:, retained[retained < len(parallel_columns)]]
    union_X = union_X[:, retained]
    walks = json.loads((ROOT / 'research' / 'chi_walk_angle_cache.json').read_text())['tables']
    for name, walk in walks.items():
        features[name]['chi_walk_rot_near_frac'] = model_features(walk)['chi_walk_rot_near_frac']
    actual = np.asarray([float(r['actual_s']) for r in rows])
    for seed in ALTERNATE_SEEDS:
        assigned = alternate_folds(row_names, seed)
        prediction = cross_predict(parallel_X, union_X, y, timeout, thresholds,
                                   row_names, training, features, assigned, 'matched')
        base = prediction['basis_adjusted_seconds']
        result = {'baseline_score': challenge_score(actual, base), 'candidates': {}}
        for set_name, config in CANDIDATES.items():
            nearest, distances, counts, log_actual = neighbour_predictions(
                distances_by_set[set_name], names, rows, assigned, 'matched', signatures)
            proposed, changed = apply_candidate(base, nearest, distances,
                                                 counts, log_actual, config)
            result['candidates'][set_name] = {
                'score': challenge_score(actual, proposed),
                'changed_rows': changed,
            }
        report['alternate_seeds'][str(seed)] = result
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print('alternate', seed, result, flush=True)


if __name__ == '__main__':
    main()
