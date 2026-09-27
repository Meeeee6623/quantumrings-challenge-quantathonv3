"""Extract, validate, and train the Quantum Rings runtime model.

Run from challenge root: .venv/bin/python research/train_runtime.py
All threshold rows of each circuit stay together in cross-validation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'quantathon-harness'))
from model import RuntimeModel
from run import read_qasm

CAP = 14400.0
CACHE = ROOT / 'research' / 'features.json'
ARTIFACT = ROOT / 'quantathon-harness' / 'artifacts' / 'runtime_model.joblib'


def extract():
    model = RuntimeModel(full_features=True)
    paths = sorted((ROOT / 'training_circuits').glob('*.qasm.zst'))
    data = {}
    slow = []
    started = time.perf_counter()
    for i, path in enumerate(paths, 1):
        qasm = read_qasm(path)
        t0 = time.perf_counter()
        data[path.name[:-4]] = model.featurize(qasm)
        elapsed = time.perf_counter() - t0
        slow.append((elapsed, path.name, len(qasm)))
        del qasm
        if i % 25 == 0 or i == len(paths):
            print(f'extracted {i}/{len(paths)} in {time.perf_counter()-started:.1f}s; slowest={max(slow)[0]:.2f}s', flush=True)
    CACHE.write_text(json.dumps(data, separators=(',', ':')))
    print('slowest', sorted(slow, reverse=True)[:10], flush=True)
    print('unsupported', sorted([(v['unsupported_statements'],k) for k,v in data.items()], reverse=True)[:8])
    return data


def score(y_true, pred, is_timeout):
    sec = np.maximum(1e-9, np.power(10.0, np.clip(pred,-9,9)))
    sec = np.where(is_timeout, np.minimum(sec,CAP), sec)
    actual = np.power(10.0, y_true)
    return np.maximum(0, 1 - np.abs(np.log10(sec/actual))/2)


def columns_for(feats, view):
    keys = sorted(set().union(*(f.keys() for f in feats.values())))
    # Features that compare raw and locally simplified structure constitute the
    # transpilation-inspired ablation. Keep name/metadata absent throughout.
    simplified = {'effective_ops','zero_angles','small_angles','cancel_pairs','rotation_folds'}
    basic = {'n_qubits','active_qubits','ops','one_q','two_q','multi_q','depth','two_depth',
             'measurements','resets','barriers','conditional','qasm_bytes'}
    if view == 'basic':
        keys = [k for k in keys if k in basic]
    elif view == 'raw':
        keys = [k for k in keys if k not in simplified]
    cols = keys + ['log_'+k for k in keys if k not in ('two_q_ratio','nonclifford_ratio',
             'largest_component_frac','depth_per_qubit','huge_fast_path')]
    return cols + ['threshold','log_threshold']


def matrix(rows, feats, columns):
    out = np.zeros((len(rows),len(columns)),dtype=np.float64)
    for i,r in enumerate(rows):
        f = feats[r['filename']]
        thr = int(r['threshold'])
        for j,k in enumerate(columns):
            if k == 'threshold': val = thr
            elif k == 'log_threshold': val = math.log2(thr)
            elif k.startswith('log_'): val = math.log1p(max(0,float(f.get(k[4:],0))))
            else: val = float(f.get(k,0))
            out[i,j] = val
    return out


def candidates():
    return {
        'ridge': lambda: make_pipeline(StandardScaler(),Ridge(alpha=30.0)),
        'extra_trees': lambda: ExtraTreesRegressor(n_estimators=240,min_samples_leaf=2,
                                                   max_features=.9,n_jobs=-1,random_state=17),
        'hist_boost': lambda: HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,
                                                            min_samples_leaf=12,learning_rate=.045,
                                                            l2_regularization=2.0,random_state=17),
        'random_forest': lambda: RandomForestRegressor(n_estimators=180,min_samples_leaf=2,
                                                       n_jobs=-1,random_state=17),
    }


def train(feats):
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in feats]
    names=np.array([r['filename'] for r in rows])
    y=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s'])) for r in rows])
    timeout=np.array([r['status']=='timeout' for r in rows])
    # Identical extracted profiles are held in the same fold even when the
    # scrambled filenames differ. The grouping uses no target values.
    signatures={name:hashlib.sha1(json.dumps(feats[name],sort_keys=True).encode()).hexdigest()
                for name in feats}
    groups=np.array([signatures[name] for name in names])
    splits=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,groups))
    results=[]
    for view in ('basic','raw','all'):
        cols=columns_for(feats,view)
        X=matrix(rows,feats,cols)
        for name,make in candidates().items():
            pred=np.zeros(len(rows))
            for train_idx,test_idx in splits:
                est=make()
                est.fit(X[train_idx],y[train_idx])
                pred[test_idx]=est.predict(X[test_idx])
            s=score(y,pred,timeout)
            row={'view':view,'model':name,'score':float(s.mean()),
                 'threshold_scores':{str(thr):float(s[[i for i,r in enumerate(rows) if int(r['threshold'])==thr]].mean()) for thr in (16,64,512)},
                 'timeout_score':float(s[timeout].mean()),
                 'median_factor_error':float(10**np.median(np.abs(pred-y)))}
            results.append(row)
            print(json.dumps(row),flush=True)
    # One final model is fitted on all available labels after model choice.
    winner=max(results,key=lambda r:r['score'])
    cols=columns_for(feats,winner['view'])
    X=matrix(rows,feats,cols)
    winner_pred=np.zeros(len(rows))
    for train_idx,test_idx in splits:
        fold_estimator=candidates()[winner['model']]()
        fold_estimator.fit(X[train_idx],y[train_idx])
        winner_pred[test_idx]=fold_estimator.predict(X[test_idx])
    winner_scores=score(y,winner_pred,timeout)
    worst_rows=[]
    for i in np.argsort(winner_scores)[:20]:
        worst_rows.append({'filename':rows[i]['filename'],'threshold':int(rows[i]['threshold']),
                           'status':rows[i]['status'],'actual_s':float(10**y[i]),
                           'predicted_s':float(10**winner_pred[i]),'score':float(winner_scores[i])})

    # Deliberately harder test: withhold coarse structural clusters, not only
    # individual circuits. These clusters are computed from permitted features.
    circuit_names=sorted(feats)
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(feats[name].get(k,0))))
                 for k in cluster_keys] for name in circuit_names])
    C=StandardScaler().fit_transform(C)
    cluster_ids=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(C)
    cluster_lookup=dict(zip(circuit_names,cluster_ids))
    stress_groups=np.array([cluster_lookup[name] for name in names])
    stress_pred=np.zeros(len(rows))
    stress_fold_scores=[]
    for train_idx,test_idx in GroupKFold(n_splits=5).split(X,y,stress_groups):
        fold_estimator=candidates()[winner['model']]()
        fold_estimator.fit(X[train_idx],y[train_idx])
        stress_pred[test_idx]=fold_estimator.predict(X[test_idx])
        stress_fold_scores.append(float(score(y[test_idx],stress_pred[test_idx],timeout[test_idx]).mean()))
    stress_score=float(score(y,stress_pred,timeout).mean())
    with (ROOT/'research'/'oof_predictions.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['filename','threshold','status','actual_s',
                                              'predicted_s','score','structural_predicted_s',
                                              'structural_score'])
        writer.writeheader()
        stress_scores=score(y,stress_pred,timeout)
        for i,row in enumerate(rows):
            writer.writerow({'filename':row['filename'],'threshold':row['threshold'],
                             'status':row['status'],'actual_s':10**y[i],
                             'predicted_s':10**winner_pred[i],'score':winner_scores[i],
                             'structural_predicted_s':10**stress_pred[i],
                             'structural_score':stress_scores[i]})
    estimator=candidates()[winner['model']]()
    estimator.fit(X,y)
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({'columns':cols,'estimator':estimator,'view':winner['view']},ARTIFACT,compress=3)
    report={'rows':len(rows),'circuits':len(set(names)),
            'distinct_feature_profiles':len(set(signatures.values())),
            'folds':5,'results':results,
            'winner':winner,'worst_grouped_rows':worst_rows,
            'structural_cluster_holdout_score':stress_score,
            'structural_cluster_fold_scores':stress_fold_scores,
            'artifact':str(ARTIFACT),'score_definition':'quantathon-harness/score.py'}
    (ROOT/'research'/'validation.json').write_text(json.dumps(report,indent=2))
    print('WINNER',json.dumps(winner),flush=True)
    print('STRUCTURAL CLUSTER HOLDOUT',stress_score,stress_fold_scores,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--extract-only',action='store_true')
    parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--refresh-unsupported',action='store_true',
                        help='recompute cached circuits that had unrecognized statements')
    args=parser.parse_args()
    if args.refresh_unsupported:
        feats=json.loads(CACHE.read_text())
        model=RuntimeModel(full_features=True)
        targets=[name for name,f in feats.items() if f['unsupported_statements']]
        for i,name in enumerate(targets,1):
            path=ROOT/'training_circuits'/(name+'.zst')
            qasm=read_qasm(path)
            t0=time.perf_counter()
            feats[name]=model.featurize(qasm)
            print(f'refreshed {i}/{len(targets)} {name}: unsupported={feats[name]["unsupported_statements"]}, parse={time.perf_counter()-t0:.2f}s',flush=True)
        CACHE.write_text(json.dumps(feats,separators=(',',':')))
    else:
        feats=json.loads(CACHE.read_text()) if args.train_only else extract()
    if not args.extract_only:
        train(feats)
