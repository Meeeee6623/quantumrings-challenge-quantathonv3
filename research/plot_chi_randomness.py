"""Generate reviewable graphs for the χ and randomness experiment.

uv run --locked --extra report python research/plot_chi_randomness.py
"""
import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from train_runtime import ROOT


def main():
    report = json.loads((ROOT / 'research/chi_randomness_ablation.json').read_text())
    by_name = {r['view']:r for r in report['results']}
    blend = next(r for r in report['blends']
                 if r['target_view']=='chi_plus_effective_peak' and r['target_weight']==.25)
    rows = [by_name[name] for name in ('baseline','chi_plus_random',
            'effective_chi_only','chi_plus_effective_peak')]
    rows += [blend,by_name['chi_plus_random_effective']]
    labels = ['Baseline', 'χ + diversity', 'Effective χ only',
              'χ + diversity +\npeak estimate', '25% effective χ\nblend',
              'χ + diversity +\nall estimates']
    x = np.arange(len(rows))
    grouped = [r['circuit']['score'] for r in rows]
    structural = [r['structural']['score'] for r in rows]
    fig, ax = plt.subplots(figsize=(11.5,5.3),layout='constrained')
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    width = .35
    blue = '#2456a6'; orange = '#dd7a26'
    ax.bar(x-width/2, grouped, width, color=blue,label='Circuit-grouped holdout')
    ax.bar(x+width/2, structural,width,color=orange,label='Structural-cluster holdout')
    for positions, values in ((x-width/2,grouped),(x+width/2,structural)):
        for xx,v in zip(positions,values):
            ax.text(xx,v+.003,f'{v:.4f}',ha='center',va='bottom',fontsize=8)
    ax.set(ylim=(0,.96),ylabel='Official score (higher is better)',
           title='Effective χ estimate: five-fold feature ablation')
    ax.set_xticks(x,labels)
    ax.grid(axis='y',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False)
    fig.savefig(ROOT/'research/chi_randomness_ablation.svg',bbox_inches='tight')
    fig.savefig(ROOT/'research/chi_randomness_ablation.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    fig.patch.set_facecolor('#f8fafc');ax.set_facecolor('#f8fafc')
    r=np.linspace(0,1,101)
    for upper,color in ((8,'#2a9d8f'),(16,blue),(32,orange)):
        ax.plot(r,upper*r,color=color,linewidth=2,label=f'χ upper bound = 2^{upper}')
        ax.plot([1],[upper],marker='o',color=color)
    ax.set(xlim=(0,1.02),ylim=(0,34),xlabel='Circuit-diversity proxy',
           ylabel='Estimated log₂ bond dimension',
           title='Defined proxy reaches each χ upper bound at diversity = 1')
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left')
    fig.savefig(ROOT/'research/effective_chi_interpolation.svg',bbox_inches='tight')
    fig.savefig(ROOT/'research/effective_chi_interpolation.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

    with (ROOT/'research/chi_randomness_oof.csv').open(newline='') as file:
        observations = [r for r in csv.DictReader(file)
                        if r['threshold']=='64' and float(r['chi_upper_peak'])>=16]
    random = np.array([float(r['randomness_proxy']) for r in observations])
    percentile = 100*(rankdata(random,method='average')-.5)/len(random)
    runtime = np.array([float(r['actual_s']) for r in observations])
    seed = np.random.default_rng(17)
    jitter = seed.normal(0,.002,size=len(random))
    edges = np.linspace(0,100,6)
    centers=[]; medians=[]; lower=[]; upper=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(percentile>=lo)&((percentile<=hi) if i==4 else (percentile<hi))
        centers.append(float(np.median(percentile[mask])))
        q=np.quantile(runtime[mask],[.25,.5,.75])
        lower.append(q[0]);medians.append(q[1]);upper.append(q[2])
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    fig.patch.set_facecolor('#f8fafc');ax.set_facecolor('#f8fafc')
    ax.scatter(percentile+100*jitter,runtime,s=10,alpha=.17,color=blue,rasterized=True,
               label='Circuits, threshold 64')
    ax.errorbar(centers,medians,
                yerr=[np.array(medians)-lower,np.array(upper)-medians],
                fmt='o-',color=orange,linewidth=2,capsize=4,markersize=6,
                label='Quintile median and interquartile range')
    ax.set_yscale('log')
    ax.set(xlim=(-2,102),ylim=(.03,20000),xlabel='Circuit-diversity percentile among these circuits',
           ylabel='Measured runtime (seconds; log scale)',
           title='The raw association is uneven, even among circuits with χ bound ≥ 2¹⁶')
    ax.axhline(14400,color='#7c3aed',linestyle='--',alpha=.65,label='Four-hour cap')
    ax.grid(axis='y',alpha=.18,which='both')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left')
    fig.savefig(ROOT/'research/randomness_runtime.svg',bbox_inches='tight')
    fig.savefig(ROOT/'research/randomness_runtime.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
