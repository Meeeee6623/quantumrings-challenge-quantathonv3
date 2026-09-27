"""External-label geometry recognizer and incremental runtime ablation.

Generate MQT Bench references with:
    uv run --with 'mqt-bench==2.3.0' python research/algorithm_geometry_probe.py --generate
Then run the lightweight challenge-label evaluation with:
    uv run --locked python research/algorithm_geometry_probe.py --evaluate

The classifier is research-only and is not loaded by the submission harness.
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
sys.path.insert(0,str(ROOT/'quantathon-harness'))
sys.path.insert(0,str(ROOT/'research'))
from model import Stats
from evaluate_geometry_families import ADDED
from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION, boot_delta
from train_runtime import CAP, candidates, columns_for, matrix, score

REFERENCE = ROOT/'research/algorithm_geometry_reference.json'
PREDICTIONS = ROOT/'research/algorithm_geometry_predictions.json'
REPORT = ROOT/'research/algorithm_geometry_probe.json'
FAMILIES = ('ghz','graphstate','qft','qaoa','randomcircuit','vqe_su2','wstate','qnn')
SIZES = (4,6,8,10,12,16,20,24,28,32,40,48,64)
GEOMETRY_KEYS = ('n_qubits','active_qubits','two_q','two_depth','unique_pairs',
                 'pair_reuse','max_degree','mean_degree','components',
                 'largest_component_frac','max_span','mean_span',
                 'graph_cutwidth_original','graph_cutwidth_rcm',
                 'graph_mean_cut_original','graph_mean_cut_rcm','graph_span_rcm',
                 'graph_rcm_gain','graph_degree_entropy','graph_density',
                 'graph_mindegree_width','chi_timeline_peak_auc',
                 'chi_timeline_capacity_auc','chi_timeline_mid_t50',
                 'chi_timeline_post_mid_sat_twoq')


def geometry_vector(f):
    """Only interaction topology, volume and timing; no gate-name counts."""
    n = max(1,float(f.get('n_qubits',0)))
    two = max(1,float(f.get('two_q',0)))
    pairs = max(1,float(f.get('unique_pairs',0)))
    return np.array([
        math.log1p(n),
        float(f.get('active_qubits',0))/n,
        math.log1p(float(f.get('two_q',0))/n),
        math.log1p(float(f.get('two_depth',0))/n),
        math.log1p(float(f.get('unique_pairs',0))/n),
        math.log1p(float(f.get('pair_reuse',0))),
        float(f.get('max_degree',0))/n,
        float(f.get('mean_degree',0))/n,
        float(f.get('components',0))/n,
        float(f.get('largest_component_frac',0)),
        float(f.get('max_span',0))/n,
        float(f.get('mean_span',0))/n,
        float(f.get('graph_cutwidth_original',0))/two,
        float(f.get('graph_cutwidth_rcm',0))/two,
        float(f.get('graph_mean_cut_original',0))/two,
        float(f.get('graph_mean_cut_rcm',0))/two,
        float(f.get('graph_span_rcm',0))/n,
        float(f.get('graph_rcm_gain',0)),
        float(f.get('graph_degree_entropy',0)),
        float(f.get('graph_density',0)),
        float(f.get('graph_mindegree_width',0))/n,
        float(f.get('chi_timeline_peak_auc',0))/n,
        float(f.get('chi_timeline_capacity_auc',0)),
        float(f.get('chi_timeline_mid_t50',0)),
        float(f.get('chi_timeline_post_mid_sat_twoq',0)),
    ],dtype=float)


def generate():
    from mqt.bench import BenchmarkLevel, get_benchmark
    from importlib.metadata import version
    rows = []
    started=time.perf_counter()
    for family in FAMILIES:
        for n in SIZES:
            variants = 3 if family in ('graphstate','qaoa','randomcircuit') else 1
            for variant in range(variants):
                qc = get_benchmark(family,BenchmarkLevel.ALG,n).decompose(reps=6)
                qubits = {q:i for i,q in enumerate(qc.qubits)}
                stats = Stats(n)
                total = len(qc.data)
                for i,instruction in enumerate(qc.data):
                    gate = instruction.operation.name.lower()
                    operands = [qubits[q] for q in instruction.qubits]
                    stats.add(gate,operands,window=min(7,8*i//max(1,total)))
                f = stats.features()
                rows.append({'family':family,'size':n,'variant':variant,
                             'features':{k:f.get(k,0) for k in GEOMETRY_KEYS}})
        print(f'{family}: {len(rows)} reference circuits so far',flush=True)
    payload = {'source':'mqt-bench','version':version('mqt-bench'),
               'abstraction_level':'algorithmic','decompose_reps':6,
               'families':FAMILIES,'sizes':SIZES,'rows':rows,
               'generation_seconds':time.perf_counter()-started}
    REFERENCE.write_text(json.dumps(payload,separators=(',',':')))
    print('wrote',REFERENCE,'rows',len(rows),'seconds',round(payload['generation_seconds'],2))


def evaluate():
    references=json.loads(REFERENCE.read_text())
    examples=[]
    seen=set()
    for row in references['rows']:
        key=(row['family'],row['size'],json.dumps(row['features'],sort_keys=True))
        if key not in seen:
            seen.add(key)
            examples.append(row)
    X=np.vstack([geometry_vector(row['features']) for row in examples])
    y=np.array([row['family'] for row in examples])
    sizes=np.array([row['size'] for row in examples])
    classifier=lambda:ExtraTreesClassifier(n_estimators=240,min_samples_leaf=2,
                                             max_features=.9,class_weight='balanced',
                                             n_jobs=-1,random_state=41)
    def cross_predict(columns):
        out=np.empty(len(y),dtype=object)
        for train,test in GroupKFold(n_splits=5).split(X,y,sizes):
            local=classifier()
            local.fit(X[train][:,columns],y[train])
            out[test]=local.predict(X[test][:,columns])
        return out
    pred=cross_predict(list(range(X.shape[1])))
    topology=cross_predict([0,1,4,6,7,8,9,10,11,16,17,18,19,20])
    volume_timing=cross_predict([0,2,3,5,12,13,14,15,21,22,23,24])
    reference_scores={'generated_rows':len(references['rows']),
                      'unique_geometry_rows':len(examples),
                      'size_grouped_accuracy':float(accuracy_score(y,pred)),
                      'size_grouped_macro_f1':float(f1_score(y,pred,average='macro')),
                      'topology_only_accuracy':float(accuracy_score(y,topology)),
                      'topology_only_macro_f1':float(f1_score(y,topology,average='macro')),
                      'volume_timing_only_accuracy':float(accuracy_score(y,volume_timing)),
                      'volume_timing_only_macro_f1':float(f1_score(y,volume_timing,average='macro')),
                      'families':list(FAMILIES),
                      'confusion_matrix':confusion_matrix(y,pred,labels=FAMILIES).tolist()}
    print('reference classification',json.dumps(reference_scores),flush=True)
    model=classifier()
    model.fit(X,y)
    features=json.loads((ROOT/'research/features.json').read_text())
    names=sorted(features)
    challenge_x=np.vstack([geometry_vector(features[name]) for name in names])
    probs=model.predict_proba(challenge_x)
    scaled=StandardScaler().fit(X)
    reference_scaled=scaled.transform(X)
    challenge_scaled=scaled.transform(challenge_x)
    reference_dist=pairwise_distances(reference_scaled)
    reference_dist[sizes[:,None]==sizes[None,:]]=np.inf
    reference_distance_95=float(np.quantile(reference_dist.min(axis=1),.95))
    challenge_distance=pairwise_distances(challenge_scaled,reference_scaled).min(axis=1)
    family_columns=['geometry_family_'+family for family in model.classes_]
    augmented={name:{**features[name],**dict(zip(family_columns,probs[i])),
                     'geometry_family_confidence':float(max(probs[i]))}
               for i,name in enumerate(names)}
    PREDICTIONS.write_text(json.dumps({name:{family:float(probs[i,j])
                 for j,family in enumerate(model.classes_)} for i,name in enumerate(names)},
                 separators=(',',':')))
    with (ROOT/'research/algorithm_geometry_guesses.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','qubits','guessed_mqt_family','confidence',
                         'scaled_reference_distance','outside_reference_95pct']
                        +['prob_'+family for family in model.classes_])
        for i,name in enumerate(names):
            writer.writerow([name,features[name]['n_qubits'],
                             model.classes_[int(np.argmax(probs[i]))],
                             float(np.max(probs[i])),float(challenge_distance[i]),
                             int(challenge_distance[i]>reference_distance_95)]
                            +[float(p) for p in probs[i]])
    joblib.dump({'model':model,'families':model.classes_,'keys':GEOMETRY_KEYS},
                ROOT/'research/algorithm_geometry_classifier.joblib',compress=3)

    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[row for row in csv.DictReader(file) if row['filename'] in features]
    row_names=np.array([r['filename'] for r in rows])
    target=np.array([math.log10(CAP if row['status']=='timeout'
                                 else float(row['duration_s'])) for row in rows])
    timeout=np.array([row['status']=='timeout' for row in rows])
    signatures={name:hashlib.sha1(json.dumps({k:v for k,v in f.items()
        if k not in ADDED | PREVIOUS_ABLATION},sort_keys=True).encode()).hexdigest()
        for name,f in features.items()}
    groups=np.array([signatures[name] for name in row_names])
    normal=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),target,groups))
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(features[name].get(k,0))))
                 for k in cluster_keys] for name in names])
    labels=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(StandardScaler().fit_transform(C))
    lookup=dict(zip(names,labels))
    stress=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),target,
        np.array([lookup[name] for name in row_names])))
    base_columns=joblib.load(ROOT/'quantathon-harness/artifacts/runtime_model.joblib')['columns']
    added_columns=family_columns+['geometry_family_confidence']
    # Mirror the standard model's raw+log transform for the new probabilities.
    all_columns=base_columns+added_columns+['log_'+key for key in added_columns]
    new_x=matrix(rows,augmented,all_columns)
    old_oof={(r['filename'],r['threshold']):r for r in
             csv.DictReader((ROOT/'research/geometry_family_oof.csv').open())}
    runtime_scores={}
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
        runtime_scores[split_name]={'current_score':float(old_score.mean()),
                                    'with_geometry_family_score':float(new_score.mean()),
                                    'delta':float(np.mean(new_score-old_score)),
                                    'bootstrap_95':boot_delta(old_score,new_score,row_names),
                                    'timeout_delta':float(np.mean((new_score-old_score)[timeout]))}
        predictions[split_name]={'old':old_pred,'new':new_pred}
        print('runtime',split_name,json.dumps(runtime_scores[split_name]),flush=True)
    report={'reference':reference_scores,
            'challenge':{'circuits':len(names),'high_confidence_0_7':int(np.sum(np.max(probs,axis=1)>=.7)),
                         'median_confidence':float(np.median(np.max(probs,axis=1))),
                         'median_scaled_distance_to_reference':float(np.median(challenge_distance)),
                         'reference_95pct_across_size_distance':reference_distance_95,
                         'beyond_reference_95pct_distance':int(np.sum(challenge_distance>reference_distance_95)),
                         'predicted_class_counts':{str(k):int(v) for k,v in zip(*np.unique(model.predict(challenge_x),return_counts=True))},
                         'reference_width_limit':max(SIZES),'challenge_above_reference_width':sum(features[name]['n_qubits']>max(SIZES) for name in names)},
            'runtime':runtime_scores,
            'interpretation':'External MQT family labels validate only on generated MQT circuits. Challenge family predictions are unverified. Their probabilities are deterministic functions of geometry and cannot carry new information beyond complete geometry; a runtime gain would be an inductive-bias effect.'}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'research/algorithm_geometry_oof.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','threshold','status','actual_s','confidence']+
            [f'{split}_{which}_pred_s' for split in ('circuit','structural') for which in ('old','new')])
        for i,row in enumerate(rows):
            confidence=augmented[row['filename']]['geometry_family_confidence']
            writer.writerow([row['filename'],row['threshold'],row['status'],10**target[i],confidence]
                +[10**predictions[split][which][i]
                  for split in ('circuit','structural') for which in ('old','new')])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.generate: generate()
    if args.evaluate: evaluate()
