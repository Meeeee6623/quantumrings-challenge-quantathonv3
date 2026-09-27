"""Fit the selected chi-walk feature view on all available training labels.

Run: uv run --locked python research/fit_chi_walk.py
"""
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'quantathon-harness'),str(ROOT/'research')]
from chi_walk import MODEL_FEATURE_NAMES, model_features, select_model_features
from train_runtime import ARTIFACT, CAP, candidates, matrix


def main():
    backup = ROOT/'research/runtime_model_before_chi_walk.joblib'
    base = joblib.load(backup if backup.exists() else ARTIFACT)
    assert 'ensemble' not in base and not any(c.startswith('chi_walk_') for c in base['columns'])
    base_cols = base['columns']
    walks = json.loads((ROOT/'research/chi_walk_cache.json').read_text())['tables']
    feats = json.loads((ROOT/'research/features.json').read_text())
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows = [r for r in csv.DictReader(file) if r['filename'] in feats]
    y = np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s']))
                  for r in rows])
    flat = {name:model_features(walks[name]) for name in feats}
    original_columns = MODEL_FEATURE_NAMES[:8]
    new = np.array([[select_model_features(flat[r['filename']],r['threshold'])[name]
                     for name in original_columns] for r in rows],dtype=float)
    X = np.column_stack((matrix(rows,feats,base_cols),new))
    estimator = candidates()['extra_trees']()
    estimator.fit(X,y)
    if not backup.exists():
        shutil.copy2(ARTIFACT,backup)
    joblib.dump({'columns':base_cols+list(original_columns),
                 'estimator':estimator,'view':'all',
                 'chi_walk_budget_s':3.0,'chi_walk_large_cutoff_bytes':40_000_000,
                 'chi_walk_rotation_tolerance_rad':-1.0},
                ARTIFACT,compress=3)
    print('fitted',len(rows),'labels',X.shape[1],'features','artifact',ARTIFACT)


if __name__ == '__main__':
    main()
