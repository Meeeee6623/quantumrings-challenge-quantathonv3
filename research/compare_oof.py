"""Compare another agent's out-of-fold predictions on the fixed matched folds.

Input CSV columns: filename, threshold, pred_duration_s (override the last
column name with --pred-column). Every labeled row must be present exactly once.
The caller must train each row's predictor without that row's circuit or its
feature-signature group; this script can check coverage, not training leakage.

uv run --locked python research/compare_oof.py other_agent_oof.csv
uv run --locked python research/compare_oof.py other_agent_stress_oof.csv --split structural
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAP=14400.0
BASE=ROOT/'research/rotation_filter_oof.csv'
LABELS=ROOT/'runtime-data.csv'
FOLDS=ROOT/'research/comparison_folds.csv'


def load_predictions(path,column):
    out={}
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            key=(row['filename'].strip(),int(float(row['threshold'])))
            if key in out:
                raise ValueError(f'duplicate prediction: {key}')
            value=float(row[column])
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'prediction must be positive and finite: {key}')
            out[key]=value
    return out


def row_score(actual,predicted,timeout):
    if timeout:
        predicted=min(predicted,CAP)
    return max(0.0,1.0-abs(math.log10(predicted/actual))/2.0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('other',type=Path,help='other agent CSV of out-of-fold predictions')
    parser.add_argument('--pred-column',default='pred_duration_s')
    parser.add_argument('--split',choices=('matched','structural'),default='matched')
    args=parser.parse_args()
    other=load_predictions(args.other,args.pred_column)
    current=load_predictions(BASE,f'filtered_plus_angles_{args.split}_pred_s')
    with LABELS.open(newline='') as file:
        labels=list(csv.DictReader(file))
    entries=[]
    for row in labels:
        key=(row['filename'],int(row['threshold']))
        timeout=row['status']=='timeout'
        if not timeout and not row['duration_s']:
            continue
        actual=CAP if timeout else float(row['duration_s'])
        entries.append((key,actual,timeout))
    required={key for key,_,_ in entries}
    missing=required-other.keys()
    if missing:
        raise ValueError(f'other agent is missing {len(missing)} labeled rows; examples: {sorted(missing)[:5]}')
    if required-current.keys():
        raise ValueError('current OOF artifact is missing labeled rows')
    cur_scores=np.array([row_score(a,current[k],t) for k,a,t in entries])
    oth_scores=np.array([row_score(a,other[k],t) for k,a,t in entries])
    names=np.array([k[0] for k,_,_ in entries])
    unique_names=np.unique(names)
    if args.split=='structural':
        with FOLDS.open(newline='') as file:
            cluster_by_name={r['filename']:int(r['structural_cluster'])
                             for r in csv.DictReader(file)}
        units=np.array([cluster_by_name[name] for name in names])
        bootstrap_unit='structural_cluster'
    else:
        units=names
        bootstrap_unit='circuit'
    unique,group=np.unique(units,return_inverse=True)
    difference=oth_scores-cur_scores
    sums=np.bincount(group,weights=difference,minlength=len(unique))
    counts=np.bincount(group,minlength=len(unique))
    rng=np.random.default_rng(41)
    draw=rng.integers(0,len(unique),size=(5000,len(unique)))
    boot=sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
    by_threshold={}
    for threshold in (16,64,512):
        mask=np.array([k[1]==threshold for k,_,_ in entries])
        by_threshold[str(threshold)]={
            'rows':int(mask.sum()),
            'current_score':float(cur_scores[mask].mean()),
            'other_score':float(oth_scores[mask].mean()),
            'delta_other_minus_current':float(difference[mask].mean()),
        }
    def misses(predictions):
        return sum(abs(math.log10((min(predictions[k],CAP) if t else predictions[k])/a))>1
                   for k,a,t in entries)
    def factor_errors(predictions):
        return np.array([10**abs(math.log10(
            (min(predictions[k],CAP) if t else predictions[k])/a))
            for k,a,t in entries])
    timeout=np.array([t for _,_,t in entries])
    cur_factor=factor_errors(current)
    oth_factor=factor_errors(other)
    print(json.dumps({
        'split':args.split,
        'rows':len(entries),
        'circuits':len(unique_names),
        'current_score':float(cur_scores.mean()),
        'other_score':float(oth_scores.mean()),
        'delta_other_minus_current':float(difference.mean()),
        'bootstrap_unit':bootstrap_unit,
        'paired_bootstrap_95':list(map(float,np.quantile(boot,[.025,.975]))),
        'timeout_rows':int(timeout.sum()),
        'current_timeout_score':float(cur_scores[timeout].mean()),
        'other_timeout_score':float(oth_scores[timeout].mean()),
        'current_median_factor_error':float(np.median(cur_factor)),
        'other_median_factor_error':float(np.median(oth_factor)),
        'current_p95_factor_error':float(np.quantile(cur_factor,.95)),
        'other_p95_factor_error':float(np.quantile(oth_factor,.95)),
        'current_tenfold_misses':misses(current),
        'other_tenfold_misses':misses(other),
        'by_threshold':by_threshold,
        'extra_other_rows':len(other.keys()-required),
    },indent=2))


if __name__=='__main__':
    main()
