"""Reproduce the feature-ablation and held-out parity figures."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'research'
report = json.loads((RESEARCH/'geometry_family_ablation.json').read_text())
by_view = {row['view']:row for row in report['results']}
views = ['current_single','timeline','geometry','family','timeline_geometry',
         'all_new','all_new_effective_peak']
labels = ['Current single','Cut timeline','Graph geometry','Algorithm patterns',
          'Timeline + geometry','All new','All new + effective χ']
baseline = {'circuit':.89210,'structural':.70050}

fig,axes = plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
for ax,split in zip(axes,('circuit','structural')):
    values = [100*(by_view[view][split]['score']-baseline[split]) for view in views]
    colors = ['#49769b' if view!='all_new' else '#d6863d' for view in views]
    ax.barh(np.arange(len(views)),values,color=colors)
    ax.axvline(0,color='#222222',linestyle='--',linewidth=1,
               label=f'Previous fitted blend {baseline[split]:.4f}')
    ax.set_yticks(np.arange(len(views)),labels)
    ax.invert_yaxis()
    ax.set_xlim((-.2,.65) if split=='circuit' else (-.3,1.55))
    ax.set_xlabel('Score change, percentage points (higher is better)')
    ax.set_title('Circuit-grouped folds' if split=='circuit' else 'Structural-cluster holdout')
    ax.grid(axis='x',alpha=.2)
    ax.legend(loc='lower right',fontsize=8)
fig.suptitle('Paired QASM feature ablations, 1,497 labeled runs')
fig.savefig(RESEARCH/'geometry_family_ablation.png',dpi=180)
fig.savefig(RESEARCH/'geometry_family_ablation.svg')

rows=list(csv.DictReader((RESEARCH/'geometry_family_oof.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
for ax,split in zip(axes,('circuit','structural')):
    actual=np.array([float(row['actual_s']) for row in rows])
    predicted=np.array([float(row[f'all_new_{split}_pred_s']) for row in rows])
    timeout=np.array([row['status']=='timeout' for row in rows])
    ax.scatter(actual[~timeout],predicted[~timeout],s=7,alpha=.22,c='#286f9b',
               rasterized=True,label='Completed')
    ax.scatter(actual[timeout],predicted[timeout],s=12,alpha=.5,c='#c46535',
               rasterized=True,label='Timed out; actual capped')
    ax.plot([.01,14400],[.01,14400],color='#222222',linestyle='--',linewidth=1)
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set_xlim(.01,20000);ax.set_ylim(.01,20000)
    ax.set_xlabel('Observed duration, seconds')
    ax.set_ylabel('Out-of-fold prediction, seconds')
    ax.set_title('Circuit-grouped' if split=='circuit' else 'Structural clusters held out')
    ax.grid(alpha=.18)
    ax.legend(loc='upper left',fontsize=8)
fig.suptitle('Runtime prediction on held-out circuits')
fig.savefig(RESEARCH/'geometry_family_parity.png',dpi=180)
fig.savefig(RESEARCH/'geometry_family_parity.svg')
