"""Plot external-family recognition and incremental runtime evidence."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'research/algorithm_geometry_probe.json').read_text())
reference=report['reference']
runtime=report['runtime']
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')

labels=['Topology only','Volume + timing','All geometry']
values=[reference['topology_only_macro_f1'],
        reference['volume_timing_only_macro_f1'],
        reference['size_grouped_macro_f1']]
axes[0].barh(labels,values,color=['#95aebc','#49769b','#d6863d'])
axes[0].set_xlim(0,1)
axes[0].set_xlabel('Macro F1 (held-out qubit sizes)')
axes[0].set_title('MQT family recognition')
axes[0].grid(axis='x',alpha=.2)

matrix=np.array(reference['confusion_matrix'])
im=axes[1].imshow(matrix,cmap='Blues',vmin=0,vmax=39)
families=reference['families']
axes[1].set_xticks(range(len(families)),families,rotation=60,ha='right',fontsize=8)
axes[1].set_yticks(range(len(families)),families,fontsize=8)
axes[1].set_xlabel('Predicted family')
axes[1].set_ylabel('Generated family')
axes[1].set_title('Size-grouped confusion matrix')
for i in range(len(families)):
    for j in range(len(families)):
        if matrix[i,j]:
            axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center',
                         color='white' if matrix[i,j]>20 else '#1c3040',fontsize=8)
fig.colorbar(im,ax=axes[1],fraction=.046,pad=.04)

splits=['circuit','structural']
names=['Circuit-grouped','Structural clusters']
delta=np.array([100*runtime[s]['delta'] for s in splits])
interval=np.array([[100*x for x in runtime[s]['bootstrap_95']] for s in splits])
yerr=np.vstack([delta-interval[:,0],interval[:,1]-delta])
axes[2].barh(names,delta,color=['#49769b','#d6863d'])
axes[2].errorbar(delta,range(2),xerr=yerr,fmt='none',color='#222222',capsize=4)
axes[2].axvline(0,color='#222222',linestyle='--',linewidth=1)
axes[2].set_xlim(-.75,.22)
axes[2].set_xlabel('Runtime-score change, percentage points')
axes[2].set_title('Value beyond current features')
axes[2].grid(axis='x',alpha=.2)
fig.suptitle('Geometry can recognize generated templates, but adds little runtime signal')
fig.savefig(ROOT/'research/algorithm_geometry_probe.png',dpi=180)
fig.savefig(ROOT/'research/algorithm_geometry_probe.svg')
