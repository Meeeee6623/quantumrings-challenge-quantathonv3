"""Fit a validated χ and randomness ablation on all available labels.

Run only after evaluate_chi_randomness.py, which provides the held-out scores.
"""
import argparse
import csv
import json
import math

import joblib
import numpy as np

from evaluate_chi_randomness import NEW, VIEWS
from train_runtime import ARTIFACT, CACHE, CAP, ROOT, candidates, columns_for, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', choices=('effective_blend_25', *VIEWS),
                        default='effective_blend_25')
    args = parser.parse_args()
    evaluation = json.loads((ROOT / 'research/chi_randomness_ablation.json').read_text())
    baseline = next(row for row in evaluation['results'] if row['view'] == 'baseline')
    if args.view == 'effective_blend_25':
        selected = next(row for row in evaluation['blends']
                        if row['target_view'] == 'chi_plus_effective_peak'
                        and row['target_weight'] == .25)
        components = [('chi_plus_random',.75),('chi_plus_effective_peak',.25)]
    else:
        selected = next(row for row in evaluation['results'] if row['view'] == args.view)
        components = [(args.view,1.0)]
    assert selected['circuit']['score'] > baseline['circuit']['score']
    assert selected['structural']['score'] > baseline['structural']['score']
    features = json.loads(CACHE.read_text())
    with (ROOT / 'runtime-data.csv').open(newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['filename'] in features]
    all_columns = columns_for(features, 'all')
    y = np.array([math.log10(CAP if row['status'] == 'timeout'
                             else float(row['duration_s'])) for row in rows])
    fitted = []
    for view, weight in components:
        columns = [column for column in all_columns
                   if (column[4:] if column.startswith('log_') and column != 'log_threshold'
                       else column) not in NEW - VIEWS[view]]
        reference = next(row for row in evaluation['results'] if row['view'] == view)
        assert len(columns) == reference['columns']
        X = matrix(rows, features, columns)
        estimator = candidates()['extra_trees']()
        estimator.fit(X, y)
        fitted.append({'view':view,'weight':weight,'columns':columns,'estimator':estimator})
    ARTIFACT.parent.mkdir(exist_ok=True)
    artifact = {'ensemble':fitted,'view':args.view} if len(fitted)>1 else fitted[0]
    joblib.dump(artifact, ARTIFACT, compress=3)
    print(json.dumps({'artifact':str(ARTIFACT),'view':args.view,
                      'rows':len(rows),'components':len(fitted),
                      'grouped_score':selected['circuit']['score'],
                      'structural_score':selected['structural']['score']}))


if __name__ == '__main__':
    main()
