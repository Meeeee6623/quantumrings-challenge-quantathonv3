"""Ablate the near-basis rotation filter and explicit angle features.

uv run --locked python research/evaluate_rotation_filter.py --extract
uv run --locked python research/evaluate_rotation_filter.py --evaluate
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'quantathon-harness'),str(ROOT/'research')]
from chi_walk import MODEL_FEATURE_NAMES, model_features, select_model_features, walk
from run import read_qasm
from evaluate_geometry_families import ADDED
from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION, boot_delta
from train_runtime import CAP, candidates, matrix, score

OLD=ROOT/'research/chi_walk_cache.json'
NEW=ROOT/'research/chi_walk_angle_cache.json'
REPORT=ROOT/'research/rotation_filter_probe.json'
OOF=ROOT/'research/rotation_filter_oof.csv'
MAX_BYTES=40_000_000
WALK_BUDGET=3.0


def extract():
    walks={};timing=[]
    paths=sorted((ROOT/'training_circuits').glob('*.qasm.zst'))
    for i,path in enumerate(paths,1):
        qasm=read_qasm(path)
        name=path.name[:-4]
        start=time.perf_counter()
        if len(qasm)>MAX_BYTES:
            walks[name]=None
            status='large_fast_path'
        else:
            try:
                walks[name]=walk(qasm,classical_tracking=True,budget_s=WALK_BUDGET)
                status='ok'
            except Exception as error:
                walks[name]=None
                status=type(error).__name__+': '+str(error)[:100]
        timing.append((name,len(qasm),time.perf_counter()-start,status))
        if i%25==0 or i==len(paths):
            print('walked',i,'/',len(paths),flush=True)
    NEW.write_text(json.dumps({'budget_s':WALK_BUDGET,'max_bytes':MAX_BYTES,
                               'rotation_tolerance_rad':0.3,'tables':walks,
                               'timing':timing},separators=(',',':')))
    print('available',sum(w is not None for w in walks.values()),
          'max seconds',max(t[2] for t in timing),flush=True)


def evaluate():
    old=json.loads(OLD.read_text())['tables']
    new=json.loads(NEW.read_text())['tables']
    feats=json.loads((ROOT/'research/features.json').read_text())
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in feats]
    names=np.array([r['filename'] for r in rows])
    y=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s'])) for r in rows])
    timeout=np.array([r['status']=='timeout' for r in rows])
    signatures={name:hashlib.sha1(json.dumps({k:v for k,v in f.items()
        if k not in ADDED | PREVIOUS_ABLATION},sort_keys=True).encode()).hexdigest()
        for name,f in feats.items()}
    groups=np.array([signatures[name] for name in names])
    circuit_names=sorted(feats)
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(feats[name].get(k,0))))
                 for k in cluster_keys] for name in circuit_names])
    labels=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(StandardScaler().fit_transform(C))
    lookup=dict(zip(circuit_names,labels))
    clusters=np.array([lookup[name] for name in names])
    splits={
        'matched':list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=17)
                       .split(np.zeros(len(rows)),clusters,groups)),
        'structural':list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,clusters)),
    }
    base_cols=joblib.load(ROOT/'research/runtime_model_before_chi_walk.joblib')['columns']
    base=matrix(rows,feats,base_cols)
    old_flat={name:model_features(w) for name,w in old.items()}
    new_flat={name:model_features(w) for name,w in new.items()}
    old_selected=[select_model_features(old_flat[r['filename']],r['threshold']) for r in rows]
    new_selected=[select_model_features(new_flat[r['filename']],r['threshold']) for r in rows]
    shape_names=MODEL_FEATURE_NAMES[:8]
    angle_names=MODEL_FEATURE_NAMES[8:]
    views={
        'baseline':base,
        'old_walk':np.column_stack((base,np.array([[v[k] for k in shape_names] for v in old_selected]))),
        'old_walk_plus_angles':np.column_stack((base,np.array([[v[k] for k in shape_names] for v in old_selected]),
                                                np.array([[v[k] for k in angle_names] for v in new_selected]))),
        'filtered_walk':np.column_stack((base,np.array([[v[k] for k in shape_names] for v in new_selected]))),
        'filtered_plus_angles':np.column_stack((base,np.array([[v[k] for k in MODEL_FEATURE_NAMES] for v in new_selected]))),
    }
    output={};predictions={}
    for view,X in views.items():
        output[view]={};predictions[view]={}
        for split_name,folds in splits.items():
            pred=np.zeros(len(rows))
            for train,test in folds:
                estimator=candidates()['extra_trees']()
                estimator.fit(X[train],y[train])
                pred[test]=estimator.predict(X[test])
            s=score(y,pred,timeout)
            entry={'score':float(s.mean()),'tenfold_misses':int(sum(abs(pred-y)>1)),
                   'threshold_scores':{str(t):float(s[[int(r['threshold'])==t for r in rows]].mean())
                                       for t in (16,64,512)}}
            output[view][split_name]=entry
            predictions[view][split_name]=pred
            print(view,split_name,json.dumps(entry),flush=True)
    for view in views:
        for split_name in splits:
            old_score=score(y,predictions['old_walk'][split_name],timeout)
            new_score=score(y,predictions[view][split_name],timeout)
            output[view][split_name]['delta_vs_old']=float((new_score-old_score).mean())
            output[view][split_name]['circuit_bootstrap_95']=boot_delta(old_score,new_score,names)
    with OOF.open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','threshold','status','actual_s']+
                        [f'{v}_{s}_pred_s' for v in views for s in splits])
        for i,r in enumerate(rows):
            writer.writerow([r['filename'],r['threshold'],r['status'],10**y[i]]+
                            [10**predictions[v][s][i] for v in views for s in splits])
    REPORT.write_text(json.dumps({'rows':len(rows),'views':output,
                                  'rotation_tolerance_rad':0.3,
                                  'features':list(MODEL_FEATURE_NAMES)},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.extract:extract()
    if args.evaluate:evaluate()
