"""Plot the extended algorithm-family experiment (uv run --locked --extra report)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'research/shor_family_probe.json').read_text())
external=report['external']
runtime=report['runtime']
fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')

names=['Topology only','Volume + timing','All geometry']
values=[external['topology_only_macro_f1'],external['volume_timing_only_macro_f1'],
        external['size_grouped_macro_f1']]
axes[0].barh(names,values,color=['#9bb3c1','#577e9d','#db873e'])
axes[0].set_xlim(0,1)
axes[0].set_xlabel('Macro F1, held-out qubit sizes')
axes[0].set_title('10 MQT algorithm families')
axes[0].grid(axis='x',alpha=.2)

shor=report['shor_out_of_family']
size=[str(r['size'])+' qubits' for r in shor]
dist=[r['scaled_reference_distance'] for r in shor]
threshold=report['challenge']['reference_95pct_distance']
axes[1].barh(size,dist,color='#a45f80')
axes[1].axvline(threshold,color='#222',linestyle='--',linewidth=1,
                label='95th percentile among reference sizes')
axes[1].set_xlabel('Nearest standardized geometry distance')
axes[1].set_title('Shor is outside reference geometry')
axes[1].legend(fontsize=8,loc='lower right')
axes[1].grid(axis='x',alpha=.2)
for i,r in enumerate(shor):
    axes[1].text(dist[i]+.08,i,f"{r['nearest_known_family']} {r['max_probability']:.2f}",
                 va='center',fontsize=8)
axes[1].set_xlim(0,max(dist)+1.35)

split=['circuit','structural']
labels=['Circuit-grouped','Structural clusters']
delta=np.array([100*runtime[s]['delta'] for s in split])
axes[2].barh(labels,delta,color=['#577e9d','#db873e'])
for i,s in enumerate(split):
    interval=runtime[s]['cluster_bootstrap_95'] if s=='structural' else runtime[s]['bootstrap_95']
    lo,hi=100*np.array(interval)
    axes[2].errorbar(delta[i],i,xerr=np.array([[delta[i]-lo],[hi-delta[i]]]),
                     fmt='none',ecolor='#222',capsize=4)
axes[2].axvline(0,color='#222',linestyle='--',linewidth=1)
axes[2].set_xlim(-.8,.9)
axes[2].set_xlabel('Runtime score change, percentage points')
axes[2].set_title('Increment beyond current model')
axes[2].grid(axis='x',alpha=.2)
fig.suptitle('QPE and arithmetic templates are recognizable; Shor and runtime transfer remain uncertain')
fig.savefig(ROOT/'research/shor_family_probe.png',dpi=180)
fig.savefig(ROOT/'research/shor_family_probe.svg')
