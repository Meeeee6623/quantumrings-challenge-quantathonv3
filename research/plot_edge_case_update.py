"""Plot large-file parser timing and fixed-fold edge-case corrections."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'research'


def submission_times(path):
    with path.open(newline='') as file:
        return {row['filename']:float(row['parse_s'])
                for row in csv.DictReader(file)}


def score(actual, predicted):
    return max(0.0,1.0-abs(math.log10(predicted/actual))/2.0)


def main():
    before = submission_times(HERE / 'training_submission_full_union.csv')
    after = submission_times(HERE / 'training_submission_final.csv')
    features = json.loads((HERE / 'features.json').read_text())
    with (HERE / 'full_union_model_oof.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    names = sorted(before.keys() & after.keys())
    x = np.asarray([features[name]['qasm_bytes'] for name in names]) / 1e6
    old = np.asarray([before[name] for name in names])
    new = np.asarray([after[name] for name in names])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    ax = axes[0]
    ax.scatter(x,old,s=14,alpha=.55,label='Earlier full-union run',color='#64748b')
    ax.scatter(x,new,s=14,alpha=.65,label='Optimized scanner',color='#0891b2')
    ax.axhline(15,color='#e11d48',linestyle='--',linewidth=1,label='Harness parse cap')
    ax.set(xlabel='Decoded QASM size (MB)',ylabel='Parse time (seconds)',
           title='All 532 circuits: parser timing')
    ax.set_xscale('log')
    ax.legend(fontsize=8,loc='upper left')

    bins = [(0,1e6,'<1 MB'),(1e6,1e7,'1–10 MB'),
            (1e7,4e7,'10–40 MB'),(4e7,math.inf,'>40 MB')]
    labels=[]; base=[]; corrected=[]; counts=[]
    for lo,hi,label in bins:
        subset=[row for row in rows
                if lo <= features[row['filename']]['qasm_bytes'] < hi]
        labels.append(label)
        counts.append(len(subset))
        base.append(np.mean([score(float(row['actual_s']),
                                   float(row['matched_full_union_pred_s']))
                             for row in subset]))
        corrected.append(np.mean([score(float(row['actual_s']),
                                        float(row['matched_template_pred_s']))
                                  for row in subset]))
    ax=axes[1]
    positions=np.arange(len(bins))
    width=.36
    ax.bar(positions-width/2,base,width,color='#64748b',label='Full union')
    ax.bar(positions+width/2,corrected,width,color='#0891b2',label='Floors + template blend')
    ax.set_xticks(positions,labels)
    ax.set_ylim(.65,1.0)
    ax.set(ylabel='Matched out-of-fold score',title='Accuracy by decoded file size')
    for i,count in enumerate(counts):
        ax.text(i,.657,f'n={count}',ha='center',fontsize=8)
        ax.text(i,corrected[i]+.004,f'+{corrected[i]-base[i]:.3f}',
                ha='center',fontsize=8,color='#0e7490')
    ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Quantum Rings: large-file and structural edge cases',weight='bold')
    fig.savefig(HERE / 'edge_case_update.png',dpi=180,bbox_inches='tight')


if __name__ == '__main__':
    main()
