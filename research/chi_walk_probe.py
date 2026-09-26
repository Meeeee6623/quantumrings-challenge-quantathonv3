"""Ablate the user-provided chi walk against the current runtime model.

uv run --locked python research/chi_walk_probe.py --extract
uv run --locked python research/chi_walk_probe.py --evaluate
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
from chi_walk import walk, price
from run import read_qasm
from evaluate_geometry_families import ADDED
from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION, boot_delta
from train_runtime import CAP, candidates, matrix, score

CACHE=ROOT/'research/chi_walk_cache.json'
REPORT=ROOT/'research/chi_walk_probe.json'
OOF=ROOT/'research/chi_walk_oof.csv'
MAX_BYTES=40_000_000
WALK_BUDGET=3.0

VIEWS={
    'cost_p2':('cost_p2',),
    'cost_p2_5':('cost_p2_5',),
    'cost_p3':('cost_p3',),
    'cost_overhead':('cost_overhead',),
    'cost_high_overhead':('cost_high_overhead',),
    'shape':('entangling_frac','max_logchi','sat_frac','sat_start','links_at_cap','extrapolated','available'),
    'shape_cost':('cost_overhead','entangling_frac','max_logchi','sat_frac','sat_start','links_at_cap','extrapolated','available'),
    'cost_uncapped':('cost_overhead','uncapped_cost','extrapolated','available'),
}
PRICING={
    'cost_p2':(0,0,2.0),
    'cost_p2_5':(0,0,2.5),
    'cost_p3':(0,0,3.0),
    'cost_overhead':(10,100,2.5),
    'cost_high_overhead':(100,1000,2.5),
}


def extract():
    paths=sorted((ROOT/'training_circuits').glob('*.qasm.zst'))
    result={}
    timing=[]
    started=time.perf_counter()
    for i,path in enumerate(paths,1):
        qasm=read_qasm(path)
        key=path.name[:-4]
        if len(qasm)>MAX_BYTES:
            result[key]=None
            timing.append((key,len(qasm),0.0,'large_fast_path'))
        else:
            t=time.perf_counter()
            try:
                result[key]=walk(qasm,classical_tracking=True,budget_s=WALK_BUDGET,
                                 rotation_tolerance=-1.0)
                status='ok'
            except (ValueError,IndexError,KeyError,TypeError) as error:
                result[key]=None
                status=type(error).__name__+': '+str(error)[:100]
            timing.append((key,len(qasm),time.perf_counter()-t,status))
        if i%25==0 or i==len(paths):
            print('walked',i,'/',len(paths),'seconds',round(time.perf_counter()-started,1),flush=True)
    payload={'budget_s':WALK_BUDGET,'max_bytes':MAX_BYTES,
             'tables':result,
             'timing':timing}
    CACHE.write_text(json.dumps(payload,separators=(',',':')))
    print('wrote',CACHE,'available',sum(w is not None for w in result.values()),
          'max walk seconds',max(x[2] for x in timing),flush=True)


def row_features(w,threshold):
    if w is None:
        return {key:0.0 for key in (*PRICING,'entangling_frac','max_logchi',
                'sat_frac','sat_start','links_at_cap','extrapolated','available','uncapped_cost')}
    key=str(threshold)
    tab=w['tables'][key]
    out={name:price(w,*config)[key] for name,config in PRICING.items()}
    out.update(entangling_frac=w['n_2q_entangling']/max(1,w['n_2q']),
               max_logchi=tab['max_logchi'],
               sat_frac=tab['sat_2q_gates']/max(1,w['n_2q']),
               sat_start=tab['sat_start_frac'],
               links_at_cap=tab['frac_links_at_cap'],
               extrapolated=w['extrapolated'],available=1.0,
               uncapped_cost=price(w,10,100,2.5)['uncapped'])
    return out


def evaluate():
    raw=json.loads(CACHE.read_text())
    walks=raw['tables']
    features=json.loads((ROOT/'research/features.json').read_text())
    assert features.keys()==walks.keys()
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in features]
    names=np.array([r['filename'] for r in rows])
    y=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s'])) for r in rows])
    timeout=np.array([r['status']=='timeout' for r in rows])
    signatures={name:hashlib.sha1(json.dumps({k:v for k,v in f.items()
        if k not in ADDED | PREVIOUS_ABLATION},sort_keys=True).encode()).hexdigest()
        for name,f in features.items()}
    groups=np.array([signatures[name] for name in names])
    circuit_names=sorted(features)
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(features[name].get(k,0))))
                 for k in cluster_keys] for name in circuit_names])
    labels=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(StandardScaler().fit_transform(C))
    lookup=dict(zip(circuit_names,labels))
    clusters=np.array([lookup[name] for name in names])
    # Keep repeated settings of a circuit together while matching the
    # observed feature-profile distribution across train and holdout.
    matched=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=17)
                 .split(np.zeros(len(rows)),clusters,groups))
    stress=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,clusters))
    row_map={(r['filename'],r['threshold']):row_features(walks[r['filename']],int(r['threshold']))
             for r in rows}
    base_cols=joblib.load(ROOT/'quantathon-harness/artifacts/runtime_model.joblib')['columns']
    base_x=matrix(rows,features,base_cols)
    splits_by_name={'matched':matched,'structural':stress}
    baseline_predictions={}
    for split_name,splits in splits_by_name.items():
        baseline=np.zeros(len(rows))
        for train,test in splits:
            est=candidates()['extra_trees']()
            est.fit(base_x[train],y[train])
            baseline[test]=est.predict(base_x[test])
        baseline_predictions[split_name]=baseline
        print('baseline',split_name,float(score(y,baseline,timeout).mean()),flush=True)
    results={}
    predictions={}
    for view,columns in VIEWS.items():
        new_x=np.column_stack((base_x,np.array([[row_map[(r['filename'],r['threshold'])][k]
            for k in columns] for r in rows])))
        results[view]={}
        predictions[view]={}
        for split_name,splits in splits_by_name.items():
            pred=np.zeros(len(rows))
            for train,test in splits:
                est=candidates()['extra_trees']()
                est.fit(new_x[train],y[train])
                pred[test]=est.predict(new_x[test])
            base_pred=baseline_predictions[split_name]
            old_score=score(y,base_pred,timeout)
            new_score=score(y,pred,timeout)
            delta=new_score-old_score
            entry={'current_score':float(old_score.mean()),
                   'with_walk_score':float(new_score.mean()),
                   'delta':float(delta.mean()),
                   'circuit_bootstrap_95':boot_delta(old_score,new_score,names),
                   'timeout_delta':float(delta[timeout].mean())}
            if split_name=='structural':
                unique=np.unique(clusters)
                sums=np.array([delta[clusters==g].sum() for g in unique])
                counts=np.array([sum(clusters==g) for g in unique])
                rng=np.random.default_rng(41)
                sampled=rng.integers(0,len(unique),size=(10000,len(unique)))
                boot=sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
                entry['cluster_bootstrap_95']=list(map(float,np.quantile(boot,[.025,.975])))
                entry['positive_clusters']=int(sum(sums>0))
            results[view][split_name]=entry
            predictions[view][split_name]=pred
            print(view,split_name,json.dumps(entry),flush=True)
    with OOF.open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','threshold','status','actual_s']+
            [f'baseline_{split}_pred_s' for split in splits_by_name]+
            [f'{view}_{split}_pred_s' for view in VIEWS for split in splits_by_name])
        for i,r in enumerate(rows):
            writer.writerow([r['filename'],r['threshold'],r['status'],10**y[i]]+
                [10**baseline_predictions[split][i] for split in splits_by_name]+
                [10**predictions[view][split][i] for view in VIEWS for split in splits_by_name])
    missing=[name for name,w in walks.items() if w is None]
    report={'rows':len(rows),'circuits':len(walks),'walk_available':len(walks)-len(missing),
            'unavailable':missing,'extrapolated':sum(w['extrapolated']>0 for w in walks.values() if w),
            'max_walk_seconds':max(x[2] for x in raw['timing']),
            'pricing':{k:dict(zip(('c1','c2','p'),v)) for k,v in PRICING.items()},
            'views':{k:list(v) for k,v in VIEWS.items()},'runtime':results,
            'splits':{'matched':'5-fold StratifiedGroupKFold by 12 feature-only structural clusters, grouped by circuit signature',
                      'structural':'5-fold GroupKFold holding out entire feature-only structural clusters'},
            'matched_fold_balance':[{'fold':i,'rows':len(test),
                'circuits':len(set(names[test])),
                'threshold_counts':{str(t):int(sum(int(rows[j]['threshold'])==t for j in test))
                                    for t in (16,64,512)},
                'cluster_counts':{str(g):int(sum(clusters[test]==g)) for g in range(12)}}
                for i,(_,test) in enumerate(matched)]}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.extract: extract()
    if args.evaluate: evaluate()
