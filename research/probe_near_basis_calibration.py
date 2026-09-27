"""Probe threshold-specific correction for circuits with near-basis rotations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'research'
sys.path[:0] = [str(ROOT / 'quantathon-harness'),str(HERE)]
from chi_walk import model_features  # noqa: E402
from train_merged_model import load_fold_table, paired_bootstrap, score  # noqa: E402


def evaluate(rows, features, near_fraction, folds, split, threshold, factor):
    before = np.asarray([float(row[f'{split}_template_pred_s']) for row in rows])
    after = before.copy()
    changed = []
    for i,row in enumerate(rows):
        name = row['filename']
        if (int(row['threshold']) == threshold
                and features[name]['ops'] >= 1000
                and near_fraction[name] >= .9
                and 1 <= before[i] < 14400):
            after[i] *= factor
            changed.append(name)
    y = np.log10([float(row['actual_s']) for row in rows])
    timeout = np.asarray([row['status'] == 'timeout' for row in rows])
    names = np.asarray([row['filename'] for row in rows])
    old_scores = score(y,np.log10(before),timeout)
    new_scores = score(y,np.log10(after),timeout)
    paired = paired_bootstrap(y,timeout,names,folds,split,before,after)
    return {'score_before':float(old_scores.mean()),
            'score_after':float(new_scores.mean()),
            'delta':float(new_scores.mean()-old_scores.mean()),
            'changed_rows':len(changed),
            'helped_rows':int((new_scores>old_scores).sum()),
            'hurt_rows':int((new_scores<old_scores).sum()),
            'changed_circuits':changed,
            'paired_bootstrap':paired}


def main():
    features = json.loads((HERE / 'features.json').read_text())
    walks = json.loads((HERE / 'chi_walk_angle_cache.json').read_text())['tables']
    near_fraction = {name:model_features(walk)['chi_walk_rot_near_frac']
                     for name,walk in walks.items()}
    with (HERE / 'full_union_model_oof.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    folds = load_fold_table()
    report = {}
    for threshold in (16,64,512):
        for factor in (.5,.33):
            key = f'threshold_{threshold}_factor_{factor}'
            report[key] = {split:evaluate(rows,features,near_fraction,folds,
                                           split,threshold,factor)
                           for split in ('matched','structural')}
            print(key,round(report[key]['matched']['delta'],6),
                  round(report[key]['structural']['delta'],6))
    (HERE / 'near_basis_calibration_probe.json').write_text(
        json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
