"""Fit the validated geometry/family feature model on all labeled circuits."""
import csv
import json
import math

import joblib
import numpy as np

from evaluate_geometry_families import ADDED, VIEWS
from train_runtime import ARTIFACT, CACHE, CAP, ROOT, candidates, columns_for, matrix


def main():
    report = json.loads((ROOT/'research/geometry_family_ablation.json').read_text())
    selected = next(row for row in report['results'] if row['view']=='all_new')
    assert selected['circuit']['score'] > .89210
    assert selected['structural']['score'] > .70050
    feats = json.loads(CACHE.read_text())
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['filename'] in feats]
    blocked_base = {'chi_random_peak','chi_random_pressure','chi_est_log2_peak',
                    'chi_est_log2_mid','chi_est_log2_mean','chi_est_capacity_fraction'}
    excluded = (ADDED-VIEWS['all_new']) | blocked_base
    cols = [c for c in columns_for(feats,'all')
            if (c[4:] if c.startswith('log_') and c != 'log_threshold' else c)
            not in excluded]
    assert len(cols)==selected['columns']
    X = matrix(rows,feats,cols)
    y = np.array([math.log10(CAP if row['status']=='timeout'
                             else float(row['duration_s'])) for row in rows])
    estimator = candidates()['extra_trees']()
    estimator.fit(X,y)
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({'columns':cols,'estimator':estimator,'view':'all_new'},
                ARTIFACT,compress=3)
    print(json.dumps({'artifact':str(ARTIFACT),'view':'all_new',
                      'rows':len(rows),'columns':len(cols),
                      'grouped_score':selected['circuit']['score'],
                      'structural_score':selected['structural']['score']}))


if __name__=='__main__':
    main()
