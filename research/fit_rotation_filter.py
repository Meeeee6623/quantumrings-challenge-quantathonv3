"""Fit the angle-aware χ walk and explicit near-0/π rotation features.

Run: uv run --locked python research/fit_rotation_filter.py
"""
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'quantathon-harness'),str(ROOT/'research')]
from chi_walk import BASIS_ROTATION_TOLERANCE, MODEL_FEATURE_NAMES, model_features, select_model_features
from train_runtime import ARTIFACT, CAP, candidates, matrix


def main():
    original=ROOT/'research/runtime_model_before_chi_walk.joblib'
    previous=ROOT/'research/runtime_model_before_rotation_filter.joblib'
    base=joblib.load(original)
    base_cols=base['columns']
    assert len(base_cols)==231 and not any(k.startswith('chi_walk_') for k in base_cols)
    walks=json.loads((ROOT/'research/chi_walk_angle_cache.json').read_text())['tables']
    feats=json.loads((ROOT/'research/features.json').read_text())
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in feats]
    y=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s']))
                for r in rows])
    flat={name:model_features(walks[name]) for name in feats}
    angle_x=np.array([[select_model_features(flat[r['filename']],r['threshold'])[name]
                       for name in MODEL_FEATURE_NAMES] for r in rows],dtype=float)
    X=np.column_stack((matrix(rows,feats,base_cols),angle_x))
    estimator=candidates()['extra_trees']()
    estimator.fit(X,y)
    if not previous.exists():
        shutil.copy2(ARTIFACT,previous)
    joblib.dump({'columns':base_cols+list(MODEL_FEATURE_NAMES),
                 'estimator':estimator,'view':'all',
                 'chi_walk_budget_s':3.0,'chi_walk_large_cutoff_bytes':40_000_000,
                 'chi_walk_rotation_tolerance_rad':BASIS_ROTATION_TOLERANCE},
                ARTIFACT,compress=3)
    print('fitted',len(rows),'labels',X.shape[1],'features','artifact',ARTIFACT)


if __name__=='__main__':
    main()
