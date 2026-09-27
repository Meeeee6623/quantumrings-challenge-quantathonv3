"""Create the presentation figure for the merged-model comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
compact = json.loads((ROOT/'research/merged_model_validation.json').read_text())
union = json.loads((ROOT/'research/full_union_model_validation.json').read_text())

names = ['Parallel global','Compact merge','Full union']
matched = [
    union['splits']['matched']['parallel_global']['score'],
    compact['splits']['matched']['merged']['score'],
    union['splits']['matched']['full_union']['score'],
]
structural = [
    union['splits']['structural']['parallel_global']['score'],
    compact['splits']['structural']['merged']['score'],
    union['splits']['structural']['full_union']['score'],
]
timeout = [
    union['splits']['matched']['parallel_global']['timeout_score'],
    compact['splits']['matched']['merged']['timeout_score'],
    union['splits']['matched']['full_union']['timeout_score'],
]
misses = [
    union['splits']['matched']['parallel_global']['tenfold_misses'],
    compact['splits']['matched']['merged']['tenfold_misses'],
    union['splits']['matched']['full_union']['tenfold_misses'],
]
colors = ['#6b7280','#38bdf8','#14b8a6']
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes = plt.subplots(1,3,figsize=(13.5,4.2),constrained_layout=True)

for index,(label,values,limits) in enumerate([
    ('Official score',matched,(.90,.93)),
    ('Structural stress score',structural,(.70,.78)),
    ('Timeout-row score',timeout,(.65,.98)),
]):
    bars = axes[index].bar(names,values,color=colors,width=.65)
    axes[index].set_ylim(*limits)
    axes[index].set_title(label,weight='bold')
    axes[index].tick_params(axis='x',rotation=18)
    for bar,value in zip(bars,values):
        axes[index].text(bar.get_x()+bar.get_width()/2,value+.002,
                         f'{value:.3f}',ha='center',va='bottom',fontsize=9,weight='bold')
    if index == 0:
        for bar,miss in zip(bars,misses):
            axes[index].text(bar.get_x()+bar.get_width()/2,limits[0]+.001,
                             f'{miss} >10× misses',ha='center',va='bottom',fontsize=8,
                             color='white',rotation=90,weight='bold')

fig.suptitle('Quantum Rings runtime predictor — fixed-fold comparison',fontsize=15,weight='bold')
fig.text(.5,-.03,'1,497 labeled runs · 532 circuits · higher score is better',
         ha='center',fontsize=10,color='#4b5563')
fig.savefig(ROOT/'research/merged_model_comparison.png',dpi=190,bbox_inches='tight')
