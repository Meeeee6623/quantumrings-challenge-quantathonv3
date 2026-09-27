"""Research-only QPE/arithmetic geometry extension and Shor out-of-family probe.

uv run --with 'mqt-bench==2.3.0' python research/shor_family_probe.py --generate
uv run --locked python research/shor_family_probe.py --evaluate
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT/'quantathon-harness')]
from algorithm_geometry_probe import GEOMETRY_KEYS, SIZES, geometry_vector
from evaluate_geometry_families import ADDED
from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION, boot_delta
from model import Stats
from train_runtime import CAP, candidates, matrix, score

OUT = ROOT/'research'
EXTRA = OUT/'shor_family_reference.json'
REPORT = OUT/'shor_family_probe.json'
GENERATORS = {
    'qpe': ('qpeexact','qpeinexact'),
    'arithmetic': ('modular_adder','multiplier','draper_qft_adder','rg_qft_multiplier'),
}


def summarize(qc):
    qubits = {q:i for i,q in enumerate(qc.qubits)}
    stats = Stats(qc.num_qubits)
    total = len(qc.data)
    for i, instruction in enumerate(qc.data):
        stats.add(instruction.operation.name.lower(),
                  [qubits[q] for q in instruction.qubits],
                  window=min(7,8*i//max(1,total)))
    f = stats.features()
    return {k:f.get(k,0) for k in GEOMETRY_KEYS}, f


def generate():
    from importlib.metadata import version
    from mqt.bench import BenchmarkLevel, get_benchmark
    rows, skipped, shor = [], [], []
    started = time.perf_counter()
    for family, names in GENERATORS.items():
        for name in names:
            for n in SIZES:
                try:
                    qc = get_benchmark(name, BenchmarkLevel.ALG, n).decompose(reps=6)
                except (ValueError, TypeError, IndexError) as error:
                    skipped.append({'generator':name,'size':n,'reason':str(error)[:150]})
                    continue
                geometry, f = summarize(qc)
                rows.append({'family':family,'generator':name,'size':n,
                             'operations':len(qc.data),'features':geometry})
            print(name, 'reference rows', len(rows), flush=True)
    # The MQT Shor generator has only four fixed widths (18, 42, 58, 74).
    # These are probes, never fitted as a supervised class. The latter two
    # expand to millions of operations and are beyond this bounded experiment.
    for n in (18,42):
        qc = get_benchmark('shor', BenchmarkLevel.ALG, n)
        geometry, f = summarize(qc)
        shor.append({'size':n,'operations':len(qc.data),'features':geometry,
                     'gate_counts':{k:int(v) for k,v in qc.count_ops().items()}})
        print('shor',n,'operations',len(qc.data),flush=True)
    payload = {'source':'mqt-bench','version':version('mqt-bench'),
               'abstraction_level':'algorithmic','decompose_reps':6,
               'shor_decompose_reps':0,'rows':rows,'shor':shor,
               'skipped':skipped,'generation_seconds':time.perf_counter()-started}
    EXTRA.write_text(json.dumps(payload,separators=(',',':')))
    print('wrote',EXTRA,'seconds',round(payload['generation_seconds'],2),flush=True)


def unique_references():
    original = json.loads((OUT/'algorithm_geometry_reference.json').read_text())
    expanded = json.loads(EXTRA.read_text())
    rows, seen = [], set()
    for row in original['rows'] + expanded['rows']:
        key = (row['family'],row['size'],json.dumps(row['features'],sort_keys=True))
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return original, expanded, rows


def evaluate():
    original, expanded, examples = unique_references()
    X = np.vstack([geometry_vector(r['features']) for r in examples])
    y = np.array([r['family'] for r in examples])
    sizes = np.array([r['size'] for r in examples])
    def classifier():
        return ExtraTreesClassifier(n_estimators=240,min_samples_leaf=2,
            max_features=.9,class_weight='balanced',n_jobs=-1,random_state=41)
    def cross_predict(columns):
        p = np.empty(len(y),dtype=object)
        for train,test in GroupKFold(n_splits=5).split(X,y,sizes):
            m=classifier()
            m.fit(X[train][:,columns],y[train])
            p[test]=m.predict(X[test][:,columns])
        return p
    pred = cross_predict(list(range(X.shape[1])))
    topology = cross_predict([0,1,4,6,7,8,9,10,11,16,17,18,19,20])
    volume = cross_predict([0,2,3,5,12,13,14,15,21,22,23,24])
    classes = sorted(set(y))
    external = {
        'generated_rows':len(original['rows'])+len(expanded['rows']),
        'unique_geometry_rows':len(examples),
        'class_counts':{k:int(np.sum(y==k)) for k in classes},
        'size_grouped_accuracy':float(accuracy_score(y,pred)),
        'size_grouped_macro_f1':float(f1_score(y,pred,average='macro')),
        'topology_only_macro_f1':float(f1_score(y,topology,average='macro')),
        'volume_timing_only_macro_f1':float(f1_score(y,volume,average='macro')),
        'per_class_f1':{k:float(f1_score(y==k,pred==k)) for k in classes},
        'classes':classes,
        'confusion_matrix':confusion_matrix(y,pred,labels=classes).tolist(),
    }
    qpe_pairs = {}
    for n in SIZES:
        a=[r for r in expanded['rows'] if r['generator']=='qpeexact' and r['size']==n]
        b=[r for r in expanded['rows'] if r['generator']=='qpeinexact' and r['size']==n]
        if a and b:
            qpe_pairs[str(n)] = a[0]['features']==b[0]['features']
    model=classifier()
    model.fit(X,y)
    scaler=StandardScaler().fit(X)
    ref_scaled=scaler.transform(X)
    dist=pairwise_distances(ref_scaled)
    dist[sizes[:,None]==sizes[None,:]]=np.inf
    distance95=float(np.quantile(dist.min(axis=1),.95))
    shor=[]
    for row in expanded['shor']:
        xv=geometry_vector(row['features']).reshape(1,-1)
        p=model.predict_proba(xv)[0]
        shor.append({'size':row['size'],'operations':row['operations'],
                     'nearest_known_family':str(model.classes_[np.argmax(p)]),
                     'max_probability':float(max(p)),
                     'all_family_probabilities':dict(zip(model.classes_,map(float,p))),
                     'scaled_reference_distance':float(pairwise_distances(scaler.transform(xv),ref_scaled).min()),
                     'outside_reference_95pct':bool(pairwise_distances(scaler.transform(xv),ref_scaled).min()>distance95)})
    print('external',json.dumps(external),flush=True)
    print('shor',json.dumps(shor),flush=True)

    features=json.loads((OUT/'features.json').read_text())
    names=sorted(features)
    challenge_x=np.vstack([geometry_vector(features[name]) for name in names])
    probs=model.predict_proba(challenge_x)
    distances=pairwise_distances(scaler.transform(challenge_x),ref_scaled).min(axis=1)
    family_columns=['geometry_family_v2_'+k for k in model.classes_]
    augmented={name:{**features[name],**dict(zip(family_columns,probs[i])),
                     'geometry_family_v2_confidence':float(max(probs[i]))}
               for i,name in enumerate(names)}
    with (OUT/'shor_family_guesses.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','qubits','nearest_mqt_family','confidence','scaled_reference_distance',
                         'outside_reference_95pct']+['prob_'+k for k in model.classes_])
        for i,name in enumerate(names):
            writer.writerow([name,features[name]['n_qubits'],model.classes_[np.argmax(probs[i])],
                max(probs[i]),distances[i],int(distances[i]>distance95),*probs[i]])
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in features]
    row_names=np.array([r['filename'] for r in rows])
    target=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s']))
                     for r in rows])
    timeout=np.array([r['status']=='timeout' for r in rows])
    signatures={name:hashlib.sha1(json.dumps({k:v for k,v in f.items()
        if k not in ADDED | PREVIOUS_ABLATION},sort_keys=True).encode()).hexdigest()
        for name,f in features.items()}
    normal=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),target,
               np.array([signatures[name] for name in row_names])))
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(features[name].get(k,0))))
                 for k in cluster_keys] for name in names])
    labels=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(StandardScaler().fit_transform(C))
    lookup=dict(zip(names,labels))
    stress=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),target,
             np.array([lookup[name] for name in row_names])))
    base_columns=joblib.load(ROOT/'quantathon-harness/artifacts/runtime_model.joblib')['columns']
    added=family_columns+['geometry_family_v2_confidence']
    all_columns=base_columns+added+['log_'+key for key in added]
    new_x=matrix(rows,augmented,all_columns)
    old_oof={(r['filename'],r['threshold']):r for r in
             csv.DictReader((OUT/'geometry_family_oof.csv').open())}
    runtime={}
    predictions={}
    for split_name,splits in (('circuit',normal),('structural',stress)):
        new_pred=np.zeros(len(rows))
        for train,test in splits:
            estimator=candidates()['extra_trees']()
            estimator.fit(new_x[train],target[train])
            new_pred[test]=estimator.predict(new_x[test])
        old_pred=np.array([math.log10(float(old_oof[(r['filename'],r['threshold'])]
            [f'all_new_{split_name}_pred_s'])) for r in rows])
        old_score=score(target,old_pred,timeout)
        new_score=score(target,new_pred,timeout)
        runtime[split_name]={'current_score':float(old_score.mean()),
            'with_qpe_arithmetic_family_score':float(new_score.mean()),
            'delta':float(np.mean(new_score-old_score)),
            'bootstrap_95':boot_delta(old_score,new_score,row_names),
            'timeout_delta':float(np.mean((new_score-old_score)[timeout]))}
        if split_name=='structural':
            cluster=np.array([lookup[name] for name in row_names])
            cluster_ids=np.unique(cluster)
            sum_delta=np.array([(new_score-old_score)[cluster==g].sum()
                                for g in cluster_ids])
            counts=np.array([np.sum(cluster==g) for g in cluster_ids])
            rng=np.random.default_rng(41)
            sampled=rng.integers(0,len(cluster_ids),size=(10000,len(cluster_ids)))
            bootstrap=sum_delta[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
            runtime[split_name]['cluster_bootstrap_95']=list(map(float,np.quantile(bootstrap,[.025,.975])))
            runtime[split_name]['positive_clusters']=int(np.sum(sum_delta>0))
            runtime[split_name]['clusters']=len(cluster_ids)
            runtime[split_name]['cluster_effects']=[{'cluster':int(g),'rows':int(c),
                'mean_delta':float(s/c)} for g,c,s in zip(cluster_ids,counts,sum_delta)]
        predictions[split_name]={'old':old_pred,'new':new_pred}
        print('runtime',split_name,json.dumps(runtime[split_name]),flush=True)
    report={'external':external,'qpe_exact_inexact_same_geometry_by_size':qpe_pairs,
            'shor_out_of_family':shor,
            'challenge':{'circuits':len(names),'high_confidence_0_7':int(np.sum(probs.max(axis=1)>=.7)),
                'median_confidence':float(np.median(probs.max(axis=1))),
                'beyond_reference_95pct_distance':int(np.sum(distances>distance95)),
                'reference_95pct_distance':distance95,
                'nearest_family_counts':{str(k):int(v) for k,v in zip(*np.unique(model.predict(challenge_x),return_counts=True))}},
            'runtime':runtime,
            'interpretation':'Shor is out of training; nearest-class probabilities are not Shor identification. QPE and arithmetic labels are externally validated MQT generator labels, not challenge ground truth.'}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'shor_family_oof.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','threshold','status','actual_s']+
            [f'{split}_{which}_pred_s' for split in ('circuit','structural') for which in ('old','new')])
        for i,r in enumerate(rows):
            writer.writerow([r['filename'],r['threshold'],r['status'],10**target[i]]+
                [10**predictions[split][which][i]
                 for split in ('circuit','structural') for which in ('old','new')])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.generate: generate()
    if args.evaluate: evaluate()
