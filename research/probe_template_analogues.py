"""Compare train-fold-only structural analogues before the selected guard.

Run: uv run --locked python research/probe_template_analogues.py
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

from train_merged_model import load_fold_table, paired_bootstrap, score


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'research'
SIGNATURES = {
    'basic':('n_qubits','ops','two_q','multi_q'),
    'gate_counts':('n_qubits','ops','two_q','multi_q','custom_calls',
                   'gate_cx','gate_rz','gate_rx','gate_ry'),
}


def evaluate(rows, features, folds, fields, split, minimum_references):
    fold_key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
    groups = defaultdict(list)
    for row in rows:
        key = (int(row['threshold']),
               tuple(features[row['filename']].get(field) for field in fields))
        groups[key].append(row)
    before = np.asarray([float(row[f'{split}_both_floors_pred_s'])
                         for row in rows])
    after = before.copy()
    changed = 0
    for i,row in enumerate(rows):
        name = row['filename']
        key = (int(row['threshold']),tuple(features[name].get(field) for field in fields))
        analogues = [reference for reference in groups[key]
                     if folds[reference['filename']][fold_key]
                     != folds[name][fold_key]]
        if len(analogues) < minimum_references:
            continue
        reference_seconds = 10**statistics.median(
            math.log10(float(reference['actual_s'])) for reference in analogues)
        after[i] = math.sqrt(before[i]*reference_seconds)
        changed += 1
    y = np.log10([float(row['actual_s']) for row in rows])
    timeout = np.asarray([row['status'] == 'timeout' for row in rows])
    names = np.asarray([row['filename'] for row in rows])
    before_scores = score(y,np.log10(before),timeout)
    after_scores = score(y,np.log10(after),timeout)
    paired = paired_bootstrap(y,timeout,names,folds,split,before,after)
    return {'eligible_rows':changed,'score_before':float(before_scores.mean()),
            'score_after':float(after_scores.mean()),
            'delta':float(after_scores.mean()-before_scores.mean()),
            'helped_rows':int((after_scores>before_scores).sum()),
            'hurt_rows':int((after_scores<before_scores).sum()),
            'paired_bootstrap':paired}


def main():
    features = json.loads((HERE / 'features.json').read_text())
    folds = load_fold_table()
    with (HERE / 'full_union_model_oof.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    report = {}
    for name,fields in SIGNATURES.items():
        for minimum in (1,2):
            key = f'{name}_min_{minimum}'
            report[key] = {split:evaluate(rows,features,folds,fields,split,minimum)
                           for split in ('matched','structural')}
            print(key,'matched',round(report[key]['matched']['delta'],6),
                  'structural',round(report[key]['structural']['delta'],6))
    (HERE / 'template_analogue_probe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
