"""Plot research-only ordered-motif detector and runtime ablation results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'research/sequence_motif_probe.json').read_text())
external=report['external']
runtime=report['runtime']
fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')

names=['QAOA-like','Shor-like']
precision=[external['qaoa']['precision'],external['shor']['precision']]
recall=[external['qaoa']['recall'],external['shor']['recall']]
x=np.arange(2)
axes[0].bar(x-.16,precision,width=.3,color='#49769b',label='Precision')
axes[0].bar(x+.16,recall,width=.3,color='#d6863d',label='Recall')
axes[0].set_xticks(x,names)
axes[0].set_ylim(0,1.12)
axes[0].set_ylabel('Rate at fixed threshold')
axes[0].set_title('Generated MQT references')
axes[0].legend(loc='lower right',fontsize=8)
axes[0].grid(axis='y',alpha=.2)
axes[0].text(0,1.055,'162 positives / 190 negatives',ha='center',fontsize=8)
axes[0].text(1,1.055,'6 positives / 346 negatives',ha='center',fontsize=8)

challenge=report['challenge']
count=[challenge['qaoa_detected_0_45'],challenge['shor_detected_0_5']]
axes[1].bar(names,count,color=['#49769b','#d6863d'])
axes[1].set_ylim(0,40)
axes[1].set_ylabel('Circuits above motif threshold')
axes[1].set_title('532 challenge circuits; labels unknown')
axes[1].grid(axis='y',alpha=.2)
for i,v in enumerate(count):
    axes[1].text(i,v+1,str(v),ha='center',fontsize=11)

views=['qaoa_score','shor_score','both_scores','combined']
labels={'qaoa_score':'QAOA score','shor_score':'Shor score',
        'both_scores':'Both scores','combined':'All motif components'}
rows=[(view,split) for view in views for split in ('circuit','structural')]
delta=np.array([100*runtime[view][split]['delta'] for view,split in rows])
colors=['#49769b' if split=='circuit' else '#d6863d' for _,split in rows]
ys=np.arange(len(rows))
axes[2].barh(ys,delta,color=colors)
for i,(view,split) in enumerate(rows):
    item=runtime[view][split]
    interval=item['cluster_bootstrap_95'] if split=='structural' else item['circuit_bootstrap_95']
    lo,hi=100*np.array(interval)
    axes[2].errorbar(delta[i],i,xerr=[[delta[i]-lo],[hi-delta[i]]],
                     fmt='none',ecolor='#222',capsize=3)
axes[2].set_yticks(ys,[labels[view]+(' · circuit' if split=='circuit' else ' · structure')
                     for view,split in rows],fontsize=8)
axes[2].invert_yaxis()
axes[2].axvline(0,color='#222',linestyle='--',linewidth=1)
axes[2].set_xlabel('Runtime score change, percentage points')
axes[2].set_title('Increment over current runtime model')
axes[2].grid(axis='x',alpha=.2)
fig.suptitle('Ordered motifs identify benchmark templates but do not improve runtime reliably')
fig.savefig(ROOT/'research/sequence_motif_probe.png',dpi=180)
fig.savefig(ROOT/'research/sequence_motif_probe.svg')
