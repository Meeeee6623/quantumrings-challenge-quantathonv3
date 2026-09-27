"""Publication-size figures for CHI_WALK_EVALUATION.md.

Run: uv run --locked --extra report python research/plot_chi_walk.py
"""
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantathon-harness'))
from chi_walk import price

report=json.loads((ROOT/'research/chi_walk_probe.json').read_text())
views=['cost_p2','cost_p2_5','cost_p3','cost_overhead','cost_high_overhead',
       'shape','shape_cost','cost_uncapped']
labels=['cost p=2','cost p=2.5','cost p=3','cost + overhead','high overhead',
        'walk shape','shape + cost','cost + uncapped']
matched=[report['runtime'][v]['matched']['delta'] for v in views]
stress=[report['runtime'][v]['structural']['delta'] for v in views]
ci=[report['runtime'][v]['matched']['circuit_bootstrap_95'] for v in views]
y=np.arange(len(views))
fig,ax=plt.subplots(figsize=(10,5.5))
ax.barh(y+0.18,matched,height=.34,color='#176B87',label='Distribution-matched holdout')
ax.barh(y-0.18,stress,height=.34,color='#D88230',label='Structural-cluster stress holdout')
ax.errorbar(matched,y+0.18,xerr=np.array([[m-l for m,(l,u) in zip(matched,ci)],
                                         [u-m for m,(l,u) in zip(matched,ci)]]),
            fmt='none',ecolor='#0D3143',capsize=3,lw=1)
ax.set_yticks(y,labels)
ax.invert_yaxis()
ax.axvline(0,color='#333333',lw=.8)
ax.set_xlabel('Official score improvement over the same-fold baseline')
ax.set_title('Incremental value of the supplied χ walk')
ax.legend(loc='lower right',frameon=False)
ax.grid(axis='x',alpha=.2)
fig.tight_layout()
fig.savefig(ROOT/'research/chi_walk_ablation.png',dpi=200)
plt.close(fig)

walks=json.loads((ROOT/'research/chi_walk_cache.json').read_text())['tables']
with (ROOT/'runtime-data.csv').open(newline='') as file:
    rows=list(csv.DictReader(file))
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
for ax,threshold in zip(axes,(16,64,512)):
    x=[]; actual=[]; timeout=[]
    for r in rows:
        if int(r['threshold'])!=threshold or walks.get(r['filename']) is None:
            continue
        x.append(price(walks[r['filename']],10,100,2.5)[str(threshold)])
        actual.append(math.log10(14400 if r['status']=='timeout' else float(r['duration_s'])))
        timeout.append(r['status']=='timeout')
    x=np.array(x); actual=np.array(actual); timeout=np.array(timeout)
    ax.scatter(x[~timeout],actual[~timeout],s=10,alpha=.32,color='#176B87')
    ax.scatter(x[timeout],actual[timeout],s=18,alpha=.75,color='#D88230',marker='^')
    ax.set_title(f'χ cap {threshold}  ·  n={len(x)}')
    ax.set_xlabel('Walk log₁₀(cost proxy)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('log₁₀(runtime seconds); timeouts = 14,400 s')
fig.suptitle('Walk cost tracks runtime, with remaining family-specific spread',y=1.02)
fig.tight_layout()
fig.savefig(ROOT/'research/chi_walk_cost_scatter.png',dpi=200,bbox_inches='tight')
plt.close(fig)
