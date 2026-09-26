"""Validation figures for the angle-aware χ walk.

Run: uv run --locked --extra report python research/plot_rotation_filter.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'research/rotation_filter_probe.json').read_text())['views']
names=['old_walk','old_walk_plus_angles','filtered_walk','filtered_plus_angles']
labels=['Original χ walk','Original + angle counts','Filtered χ walk','Filter + angle counts']
matched=[report[n]['matched']['score'] for n in names]
stress=[report[n]['structural']['score'] for n in names]
misses=[report[n]['matched']['tenfold_misses'] for n in names]
fig,(left,right)=plt.subplots(1,2,figsize=(10.5,4.5))
x=np.arange(len(names))
left.bar(x-.18,matched,width=.35,color='#176B87',label='Distribution-matched')
left.bar(x+.18,stress,width=.35,color='#D88230',label='Structural stress')
left.set_ylim(.68,.94)
left.set_ylabel('Official out-of-fold score')
left.set_xticks(x,labels,rotation=20,ha='right')
left.set_title('Rotation filter and explicit counts')
left.grid(axis='y',alpha=.2)
left.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,fontsize=8)
right.bar(x,misses,color=['#79818B','#176B87','#176B87','#176B87'])
right.set_ylim(0,65)
right.set_xticks(x,labels,rotation=20,ha='right')
right.set_ylabel('Predictions off by >10×')
right.set_title('Large misses on matched folds')
right.grid(axis='y',alpha=.2)
for i,count in enumerate(misses):
    right.text(i,count+1,str(count),ha='center',fontsize=9)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(ROOT/'research/rotation_filter_ablation.png',dpi=200)
plt.close(fig)

with (ROOT/'research/rotation_filter_oof.csv').open(newline='') as file:
    rows=[r for r in csv.DictReader(file) if r['threshold']=='512']
actual=np.array([float(r['actual_s']) for r in rows])
old=np.array([float(r['old_walk_matched_pred_s']) for r in rows])
new=np.array([float(r['filtered_plus_angles_matched_pred_s']) for r in rows])
fig,axes=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
for ax,pred,title in zip(axes,(old,new),('Original χ walk','Filter + angle counts')):
    ax.scatter(actual,pred,s=13,alpha=.45,color='#176B87')
    ax.plot([.1,1e5],[.1,1e5],color='#333333',lw=1,ls='--')
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlim(.1,1e5);ax.set_ylim(.1,1e5)
    ax.set_title(title)
    ax.set_xlabel('Actual seconds at χ=512')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Out-of-fold predicted seconds')
for ax,pred in zip(axes,(old,new)):
    for i,r in enumerate(rows):
        if r['filename'] in ('70d5d461.qasm','8bd576fc.qasm'):
            ax.annotate(r['filename'][:8],(actual[i],pred[i]),xytext=(5,5),
                        textcoords='offset points',fontsize=7)
fig.tight_layout()
fig.savefig(ROOT/'research/rotation_filter_predictions.png',dpi=200)
