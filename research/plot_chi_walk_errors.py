"""Plot error concentrations and angle sensitivity of QAOA-like circuits.

Run: uv run --locked --extra report python research/plot_chi_walk_errors.py
"""
import csv
import json
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantathon-harness'))
from model import angle_value
from run import read_qasm

features=json.loads((ROOT/'research/features.json').read_text())
with (ROOT/'research/chi_walk_oof.csv').open(newline='') as file:
    rows=list(csv.DictReader(file))


def family(d):
    if d.get('fingerprint_qaoa',0)>=.95:
        return 'QAOA-like'
    if d.get('fingerprint_grover',0)>=.95:
        return 'Grover-like'
    if d.get('fingerprint_phase_dyadic',0)>=.8 and d.get('custom_definitions',0)>0:
        return 'Custom arithmetic'
    if d.get('fingerprint_qft',0)>=.75:
        return 'QFT-like'
    return 'Other'


labels=['QAOA-like','Grover-like','Custom arithmetic','QFT-like','Other']
total={k:0 for k in labels}
miss={k:0 for k in labels}
for row in rows:
    group=family(features[row['filename']])
    actual=14400 if row['status']=='timeout' else float(row['actual_s'])
    predicted=float(row['shape_cost_matched_pred_s'])
    total[group]+=1
    miss[group]+=abs(math.log10(predicted/actual))>1

names=[name for name,f in features.items() if family(f)=='QAOA-like']
points=[]
for name in names:
    qasm=read_qasm(ROOT/'training_circuits'/(name+'.zst'))
    rx=[angle_value(v) for v in re.findall(r'\brx\(([^)]*)\)',qasm)]
    basis_distance=np.mean([abs(math.sin(angle)) for angle in rx])
    row=next(r for r in rows if r['filename']==name and r['threshold']=='512')
    actual=float(row['actual_s'])
    predicted=float(row['shape_cost_matched_pred_s'])
    points.append((name,basis_distance,actual,abs(math.log10(predicted/actual))>1))

fig,(left,right)=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={'width_ratios':[1,1.15]})
positions=np.arange(len(labels))
rates=[100*miss[k]/total[k] for k in labels]
left.barh(positions,rates,color=['#BD5039','#BD5039','#BD5039','#BD5039','#176B87'])
left.set_yticks(positions,labels)
left.invert_yaxis()
left.set_xlim(0,48)
left.set_xlabel('Rows predicted >10× high or low (%)')
left.set_title('Large misses cluster in circuit families')
left.grid(axis='x',alpha=.2)
for i,k in enumerate(labels):
    left.text(rates[i]+.8,i,f'{miss[k]}/{total[k]}',va='center',fontsize=9)

for is_miss,color,label in [(False,'#176B87','Within 10×'),(True,'#BD5039','>10× miss')]:
    subset=[p for p in points if p[3]==is_miss]
    right.scatter([p[1] for p in subset],[p[2] for p in subset],
                  color=color,s=55,alpha=.85,label=label)
right.set_yscale('log')
right.set_ylim(4,5500)
right.set_xlabel('Mean |sin(RX angle)|; 0 ≈ basis-preserving')
right.set_ylabel('Measured runtime at χ=512 (seconds)')
rho=spearmanr([p[1] for p in points],[math.log10(p[2]) for p in points]).statistic
right.set_title(f'QAOA angle sensitivity · 15 circuits · Spearman ρ={rho:.2f}')
right.grid(alpha=.2)
right.legend(frameon=False,loc='upper left')
for name,x,y,_ in points:
    if name in {'70d5d461.qasm','8bd576fc.qasm'}:
        right.annotate(name[:8],(x,y),xytext=(6,6),textcoords='offset points',fontsize=8)
fig.tight_layout()
fig.savefig(ROOT/'research/chi_walk_error_patterns.png',dpi=200)
