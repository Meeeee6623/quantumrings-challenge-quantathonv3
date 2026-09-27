"""Benchmark a fold-isolated JEPA embedding against the selected v7 model."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'quantathon-harness'), str(ROOT / 'research')]
from feature_pruning import importance_choice, redundant_columns  # noqa: E402
from train_full_union_model import (  # noqa: E402
    CLASSIFIER_LIMIT, GLOBAL_LIMIT, SPECIALIST_LIMIT, SPECIALIST_WEIGHT, TIMEOUT_CUTOFF,
    build_template_bank, floor_fold_predictions, large_work_predictions,
    load_extended_matrix, load_fold_table, load_training_table, metrics,
    near_basis_predictions, reset_family_references, template_fold_predictions,
)
from train_merged_model import CAP, THRESHOLDS, global_regressor, positive_probability, threshold_regressor, timeout_classifier  # noqa: E402
from train_jepa import embeddings_for, fit_encoder, load_tokens  # noqa: E402

REPORT = ROOT / 'jepa' / 'results.json'
OOF = ROOT / 'jepa' / 'results_oof.csv'


def build_inputs():
    rows, parallel, parallel_columns, y, timeout, thresholds, names = load_training_table()
    features = json.loads((ROOT / 'research' / 'features.json').read_text())
    extended, extended_columns = load_extended_matrix(rows, thresholds)
    union = np.column_stack((parallel, extended))
    union_columns = parallel_columns + extended_columns
    retained, _, _ = redundant_columns(union, union_columns)
    p_indices = retained[retained < len(parallel_columns)]
    return (rows, parallel[:, p_indices], [parallel_columns[i] for i in p_indices],
            union[:, retained], [union_columns[i] for i in retained], y, timeout, thresholds, names, features)


def postprocess(seconds, rows, features, assigned):
    adjusted = floor_fold_predictions(seconds, rows, features, assigned)
    adjusted = large_work_predictions(adjusted, rows, features)
    adjusted = template_fold_predictions(adjusted, rows, features, assigned)
    return near_basis_predictions(adjusted, rows, features)


def evaluate_variant(mode, split, inputs, tokens, epochs=6):
    rows, parallel, pcols, union, ucols, y, timeout, thresholds, names, features = inputs
    folds = load_fold_table()
    fold_key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
    assigned = np.asarray([int(folds[name][fold_key]) for name in names])
    global_log = np.zeros(len(y)); specialist_log = np.zeros(len(y)); timeout_probability = np.zeros(len(y))
    training_log = []
    for fold in sorted(set(assigned)):
        train, test = assigned != fold, assigned == fold
        train_names = sorted(set(names[train]))
        encoder, fit_log = fit_encoder(tokens, train_names, epochs=epochs, seed=170 + fold)
        fit_log['fold'] = int(fold)
        training_log.append(fit_log)
        # Crucially, a fold's tree sees one coordinate system: both its train
        # and test circuits are embedded by that fold's QASM-only encoder.
        embed = embeddings_for(encoder, tokens, names, fit_log['device'])
        Z = np.asarray([embed[name] for name in names], dtype=float)
        P = np.column_stack((parallel, Z)) if mode == 'full' else parallel
        U = np.column_stack((union, Z))
        p_columns = pcols + ([f'jepa_{i:02d}' for i in range(Z.shape[1])] if mode == 'full' else [])
        u_columns = ucols + [f'jepa_{i:02d}' for i in range(Z.shape[1])]
        g_idx = importance_choice(global_regressor(17), P[train], y[train], GLOBAL_LIMIT)
        global_model = global_regressor(17).fit(P[train][:, g_idx], y[train])
        global_log[test] = global_model.predict(P[test][:, g_idx])
        for threshold in THRESHOLDS:
            train_t, test_t = train & (thresholds == threshold), test & (thresholds == threshold)
            s_idx = importance_choice(threshold_regressor(17 + fold), U[train_t], y[train_t], SPECIALIST_LIMIT)
            specialist = threshold_regressor(17 + fold).fit(U[train_t][:, s_idx], y[train_t])
            specialist_log[test_t] = specialist.predict(U[test_t][:, s_idx])
            c_idx = importance_choice(timeout_classifier(17 + fold), U[train_t], timeout[train_t].astype(int), CLASSIFIER_LIMIT)
            classifier = timeout_classifier(17 + fold).fit(U[train_t][:, c_idx], timeout[train_t].astype(int))
            timeout_probability[test_t] = positive_probability(classifier, U[test_t][:, c_idx])
    seconds = np.power(10.0, np.clip((1 - SPECIALIST_WEIGHT) * global_log + SPECIALIST_WEIGHT * specialist_log, -9, 9))
    seconds = np.where(timeout_probability >= TIMEOUT_CUTOFF, CAP, seconds)
    return postprocess(seconds, rows, features, assigned), training_log


def main():
    started = time.perf_counter()
    tokens = load_tokens()
    inputs = build_inputs()
    rows, *_rest, y, timeout, thresholds, names, _features = inputs
    baseline = json.loads((ROOT / 'research' / 'full_union_model_validation.json').read_text())
    report = {'method':'fold-isolated QASM-only masked latent prediction; tree schema and postprocessing match v7',
              'baseline':{'matched':baseline['splits']['matched']['near_basis_calibration'],
                          'structural':baseline['splits']['structural']['near_basis_calibration']},
              'acceptance_gate':{'min_structural_gain':0.005, 'max_matched_drop':0.002,
                                 'rule':'promote only if both conditions hold'}, 'variants':{}}
    oof = {'filename':names, 'threshold':thresholds}
    for mode in ('experts_only', 'full'):
        report['variants'][mode] = {}
        for split in ('matched', 'structural'):
            seconds, training = evaluate_variant('full' if mode == 'full' else 'experts', split, inputs, tokens)
            result = metrics(y, timeout, thresholds, seconds)
            base_score = report['baseline'][split]['score']
            result['delta_vs_v7'] = result['score'] - base_score
            result['encoder_training'] = training
            report['variants'][mode][split] = result
            oof[f'{mode}_{split}_pred_s'] = seconds
            print(mode, split, json.dumps({'score':result['score'], 'delta':result['delta_vs_v7']}, indent=2), flush=True)
    chosen = report['variants']['full']
    report['decision'] = ('promote' if chosen['structural']['delta_vs_v7'] >= .005
                          and chosen['matched']['delta_vs_v7'] >= -.002 else 'reject_keep_v7')
    report['elapsed_s'] = time.perf_counter() - started
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    with OOF.open('w', newline='') as file:
        fields = list(oof)
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for i in range(len(names)):
            writer.writerow({key:(value[i] if hasattr(value, '__len__') and not isinstance(value, str) else value)
                             for key, value in oof.items()})
    print(json.dumps({'report':str(REPORT), 'decision':report['decision'], 'elapsed_s':report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
