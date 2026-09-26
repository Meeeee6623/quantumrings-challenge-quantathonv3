"""Export the exact distribution-matched folds used by the current ablation.

Run: uv run --locked python research/export_comparison_folds.py
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from evaluate_geometry_families import ADDED
from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION


def main():
    features=json.loads((ROOT/'research/features.json').read_text())
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in features]
    names=np.array([r['filename'] for r in rows])
    signatures={name:hashlib.sha1(json.dumps({k:v for k,v in f.items()
        if k not in ADDED | PREVIOUS_ABLATION},sort_keys=True).encode()).hexdigest()
        for name,f in features.items()}
    groups=np.array([signatures[name] for name in names])
    circuit_names=sorted(features)
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(features[name].get(k,0))))
                 for k in cluster_keys] for name in circuit_names])
    labels=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(
        StandardScaler().fit_transform(C))
    clusters=dict(zip(circuit_names,labels))
    strata=np.array([clusters[name] for name in names])
    fold_by_name={}
    for fold,(_,test) in enumerate(StratifiedGroupKFold(
            n_splits=5,shuffle=True,random_state=17).split(
                np.zeros(len(rows)),strata,groups)):
        for name in names[test]:
            if name in fold_by_name and fold_by_name[name]!=fold:
                raise AssertionError(f'circuit split across folds: {name}')
            fold_by_name[name]=fold
    stress_by_name={}
    for fold,(_,test) in enumerate(GroupKFold(n_splits=5).split(
            np.zeros(len(rows)),np.zeros(len(rows)),strata)):
        for name in names[test]:
            if name in stress_by_name and stress_by_name[name]!=fold:
                raise AssertionError(f'circuit split across stress folds: {name}')
            stress_by_name[name]=fold
    assert len(fold_by_name)==len(features)==532
    path=ROOT/'research/comparison_folds.csv'
    with path.open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','matched_fold','structural_stress_fold',
                         'structural_cluster','feature_signature_sha1'])
        for name in circuit_names:
            writer.writerow([name,fold_by_name[name],stress_by_name[name],
                             int(clusters[name]),signatures[name]])
    print(path,'fold circuits',[
        sum(fold_by_name[name]==i for name in circuit_names) for i in range(5)])


if __name__=='__main__':
    main()
