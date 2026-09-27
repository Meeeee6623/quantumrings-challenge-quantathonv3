"""Bounded research experiment for ordered Shor and QAOA circuit motifs.

uv run --with 'mqt-bench==2.3.0' python research/sequence_motif_probe.py --generate
uv run --locked python research/sequence_motif_probe.py --extract
uv run --locked python research/sequence_motif_probe.py --evaluate
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'quantathon-harness'),str(ROOT/'research')]
from model import RuntimeModel, Stats
from run import read_qasm
from evaluate_geometry_families import ADDED
from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION, boot_delta
from train_runtime import CAP, candidates, matrix, score

OUT=ROOT/'research'
REFERENCE=OUT/'sequence_motif_reference.json'
CHALLENGE=OUT/'sequence_motif_features.json'
REPORT=OUT/'sequence_motif_probe.json'
MOTIF_KEYS=('motif_qaoa_cycle1_fraction','motif_qaoa_cycle2_fraction',
    'motif_qaoa_mixer_coverage','motif_qaoa_cost_reuse',
    'motif_qaoa_single_layer_score','motif_qaoa_score',
    'motif_cx_z_cx_ratio','motif_shor_initial_h_fraction',
    'motif_shor_tail_cp_fraction','motif_shor_tail_overlap',
    'motif_shor_tail_register_phase_fraction',
    'motif_shor_arithmetic_density','motif_shor_score')


def circuit_features(qc):
    idx={q:i for i,q in enumerate(qc.qubits)}
    stats=Stats(qc.num_qubits,sequence=True)
    total=len(qc.data)
    for i,ins in enumerate(qc.data):
        try:
            params=tuple(float(v) for v in ins.operation.params)
        except (ValueError,TypeError):
            params=()
        stats.add(ins.operation.name.lower(),[idx[q] for q in ins.qubits],
                  window=min(7,8*i//max(1,total)),parameters=params)
    f=stats.features()
    return {key:f[key] for key in MOTIF_KEYS}


def generate():
    from importlib.metadata import version
    from mqt.bench import BenchmarkLevel, get_benchmark
    from mqt.bench.benchmarks import qaoa, shor
    rows=[]
    started=time.perf_counter()
    sizes=(4,6,8,10,12,16,20,24,32)
    for n in sizes:
        for reps in (1,2,3):
            for seed in (3,10,17):
                qc=qaoa.create_circuit(n,repetitions=reps,seed=seed)
                rng=np.random.default_rng(1000+n*100+reps*10+seed)
                qc=qc.assign_parameters({p:float(rng.uniform(.2,2.9)) for p in qc.parameters})
                for form,circuit in (('native',qc),('u_cx',qc.decompose(reps=6))):
                    rows.append({'family':'qaoa','generator':'qaoa','size':n,
                        'repetitions':reps,'seed':seed,'form':form,
                        'operations':len(circuit.data),'features':circuit_features(circuit)})
        print('qaoa size',n,'rows',len(rows),flush=True)
    negatives=('ghz','graphstate','qft','qpeexact','qpeinexact',
               'modular_adder','multiplier','randomcircuit','vqe_su2','qnn','grover')
    for name in negatives:
        for n in sizes:
            if name=='grover' and n>10:
                continue
            try:
                qc=get_benchmark(name,BenchmarkLevel.ALG,n)
            except (ValueError,TypeError,IndexError) as error:
                continue
            for form,circuit in (('native',qc),('u_cx',qc.decompose(reps=6))):
                rows.append({'family':name,'generator':name,'size':n,
                             'form':form,'operations':len(circuit.data),
                             'features':circuit_features(circuit)})
        print(name,'rows',len(rows),flush=True)
    # Vary the factored integer and width instead of fitting the four fixed
    # MQT catalog examples. One decomposition pass exposes the inverse QFT.
    for number,a in ((7,2),(15,4),(21,2),(35,2),(85,2),(821,4)):
        qc=shor.Shor().construct_circuit(number,a).decompose(reps=1)
        rows.append({'family':'shor','generator':'shor','number':number,
                     'size':qc.num_qubits,'form':'u_cx',
                     'operations':len(qc.data),'features':circuit_features(qc)})
        print('shor',number,'qubits',qc.num_qubits,'operations',len(qc.data),flush=True)
    payload={'source':'mqt-bench','version':version('mqt-bench'),
             'rows':rows,'generation_seconds':time.perf_counter()-started}
    REFERENCE.write_text(json.dumps(payload,separators=(',',':')))
    print('wrote',REFERENCE,'rows',len(rows),'seconds',round(payload['generation_seconds'],2),flush=True)


def extract():
    model=RuntimeModel(full_features=True)
    paths=sorted((ROOT/'training_circuits').glob('*.qasm.zst'))
    rows={}
    started=time.perf_counter()
    for i,path in enumerate(paths,1):
        qasm=read_qasm(path)
        f=model.featurize(qasm,include_sequence=True)
        rows[path.name[:-4]]={key:f.get(key,0) for key in MOTIF_KEYS}
        if i%25==0 or i==len(paths):
            print('extracted',i,'/',len(paths),'seconds',round(time.perf_counter()-started,1),flush=True)
    CHALLENGE.write_text(json.dumps(rows,separators=(',',':')))


def binary_stats(rows,which,threshold):
    actual=np.array([r['family']==which for r in rows])
    value=np.array([r['features'][f'motif_{which}_score'] for r in rows])
    detected=value>=threshold
    tp=int(np.sum(detected & actual))
    fp=int(np.sum(detected & ~actual))
    fn=int(np.sum(~detected & actual))
    tn=int(np.sum(~detected & ~actual))
    return {'positive':int(sum(actual)),'negative':int(sum(~actual)),
            'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),
            'specificity':tn/max(1,tn+fp),
            'roc_auc':float(roc_auc_score(actual,value)),
            'average_precision':float(average_precision_score(actual,value))}


def runtime_evaluation(features,motifs):
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows=[r for r in csv.DictReader(file) if r['filename'] in features]
    row_names=np.array([r['filename'] for r in rows])
    y=np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s'])) for r in rows])
    timeout=np.array([r['status']=='timeout' for r in rows])
    signatures={name:hashlib.sha1(json.dumps({k:v for k,v in f.items()
        if k not in ADDED | PREVIOUS_ABLATION},sort_keys=True).encode()).hexdigest()
        for name,f in features.items()}
    normal=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,
                np.array([signatures[name] for name in row_names])))
    names=sorted(features)
    cluster_keys=('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                  'conditional','custom_definitions','qasm_bytes')
    C=np.array([[math.log1p(max(0,float(features[name].get(k,0))))
                 for k in cluster_keys] for name in names])
    labels=KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(StandardScaler().fit_transform(C))
    lookup=dict(zip(names,labels))
    cluster=np.array([lookup[name] for name in row_names])
    stress=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,cluster))
    augmented={name:{**features[name],**motifs[name]} for name in names}
    base_cols=joblib.load(ROOT/'quantathon-harness/artifacts/runtime_model.joblib')['columns']
    old={(r['filename'],r['threshold']):r for r in
         csv.DictReader((OUT/'geometry_family_oof.csv').open())}
    views={'qaoa_score':('motif_qaoa_score',),
           'shor_score':('motif_shor_score',),
           'both_scores':('motif_qaoa_score','motif_shor_score'),
           'qaoa':tuple(k for k in MOTIF_KEYS if 'qaoa' in k or k=='motif_cx_z_cx_ratio'),
           'shor':tuple(k for k in MOTIF_KEYS if 'shor' in k),
           'combined':MOTIF_KEYS}
    results={}
    predictions={}
    for view,keys in views.items():
        cols=base_cols+list(keys)+['log_'+k for k in keys]
        X=matrix(rows,augmented,cols)
        results[view]={}
        predictions[view]={}
        for split_name,splits in (('circuit',normal),('structural',stress)):
            p=np.zeros(len(rows))
            for train,test in splits:
                est=candidates()['extra_trees']()
                est.fit(X[train],y[train])
                p[test]=est.predict(X[test])
            old_pred=np.array([math.log10(float(old[(r['filename'],r['threshold'])]
                [f'all_new_{split_name}_pred_s'])) for r in rows])
            old_score=score(y,old_pred,timeout)
            new_score=score(y,p,timeout)
            d=new_score-old_score
            result={'current_score':float(old_score.mean()),'with_motif_score':float(new_score.mean()),
                'delta':float(d.mean()),'circuit_bootstrap_95':boot_delta(old_score,new_score,row_names)}
            if split_name=='structural':
                group=np.unique(cluster)
                sums=np.array([d[cluster==g].sum() for g in group])
                counts=np.array([np.sum(cluster==g) for g in group])
                rng=np.random.default_rng(41)
                idx=rng.integers(0,len(group),size=(10000,len(group)))
                boot=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
                result['cluster_bootstrap_95']=list(map(float,np.quantile(boot,[.025,.975])))
                result['positive_clusters']=int(np.sum(sums>0))
                result['cluster_effects']=[{'cluster':int(g),'rows':int(c),
                    'mean_delta':float(s/c)} for g,c,s in zip(group,counts,sums)]
            results[view][split_name]=result
            predictions[view][split_name]=p
            print('runtime',view,split_name,json.dumps(result),flush=True)
    with (OUT/'sequence_motif_oof.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['filename','threshold','status','actual_s']+
            [f'{view}_{split}_pred_s' for view in views for split in ('circuit','structural')])
        for i,r in enumerate(rows):
            writer.writerow([r['filename'],r['threshold'],r['status'],10**y[i]]+
                [10**predictions[view][split][i] for view in views for split in ('circuit','structural')])
    return results


def evaluate():
    refs=json.loads(REFERENCE.read_text())['rows']
    external={'qaoa':binary_stats(refs,'qaoa',.45),
              'shor':binary_stats(refs,'shor',.5)}
    for form in ('native','u_cx'):
        sample=[r for r in refs if r['form']==form and r['family']!='shor']
        external['qaoa_'+form]=binary_stats(sample,'qaoa',.45)
    for reps in (1,2,3):
        sample=[r for r in refs if r.get('repetitions')==reps or r['family']!='qaoa']
        external['qaoa_repetitions_'+str(reps)]=binary_stats(sample,'qaoa',.45)
    # A related but deliberately simple QPE hypothesis: phase-register H
    # preparation, late phase operations, and little arithmetic volume.
    # Report it as a negative/diagnostic control rather than promote it.
    external['qpe_simple_motif_auc']={}
    for form in ('native','u_cx'):
        sample=[r for r in refs if r['form']==form and r['family']!='shor']
        actual=np.array([r['family'] in ('qpeexact','qpeinexact') for r in sample])
        value=np.array([r['features']['motif_shor_initial_h_fraction']*
            r['features']['motif_shor_tail_register_phase_fraction']*
            (1-r['features']['motif_shor_arithmetic_density']) for r in sample])
        external['qpe_simple_motif_auc'][form]=float(roc_auc_score(actual,value))
    external['shor_cases']=[{'number':r['number'],'size':r['size'],
        'operations':r['operations'],'score':r['features']['motif_shor_score']}
        for r in refs if r['family']=='shor']
    print('external',json.dumps(external),flush=True)
    features=json.loads((OUT/'features.json').read_text())
    motifs=json.loads(CHALLENGE.read_text())
    assert features.keys()==motifs.keys()
    challenge={'circuits':len(motifs),
        'qaoa_detected_0_45':int(sum(f['motif_qaoa_score']>=.45 for f in motifs.values())),
        'shor_detected_0_5':int(sum(f['motif_shor_score']>=.5 for f in motifs.values())),
        'qaoa_max':max(f['motif_qaoa_score'] for f in motifs.values()),
        'shor_max':max(f['motif_shor_score'] for f in motifs.values())}
    runtime=runtime_evaluation(features,motifs)
    REPORT.write_text(json.dumps({'external':external,'challenge':challenge,'runtime':runtime,
        'interpretation':'Generated MQT cases validate motifs only for tested templates. Challenge algorithms have no labels; runtime holdouts test incremental predictive utility.'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.generate: generate()
    if args.extract: extract()
    if args.evaluate: evaluate()
