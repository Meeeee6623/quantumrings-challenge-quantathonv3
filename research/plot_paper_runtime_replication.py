"""Visualize runtime-only paper ablations and seed sensitivity.

    uv run --locked --extra report python research/plot_paper_runtime_replication.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research'
main = json.loads((OUT / 'paper_runtime_replication.json').read_text())
robust_path = OUT / 'paper_runtime_robustness.json'
robust = json.loads(robust_path.read_text()) if robust_path.exists() else None

labels = ('mlp_basic', 'mlp_graph', 'family_concat',
          'family_film_residual', 'current_v7')
display = ('MLP basic', 'MLP + graph', 'Family concat',
           'Family FiLM + residual', 'Current v7')
colors = ('#7699ad', '#517dbe', '#d49557', '#7551aa', '#269077')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
for ax, split, title in zip(axes, ('matched', 'structural'),
                            ('Distribution-matched folds', 'Structural-cluster stress folds')):
    values = [main['splits'][split][key]['score'] for key in labels]
    ax.barh(display, values, color=colors)
    ax.invert_yaxis()
    ax.set(title=title, xlabel='Mean challenge runtime score',
           xlim=(0, 1.04))
    for i, value in enumerate(values):
        ax.text(value + .001, i, f'{value:.4f}', va='center', fontsize=9)
fig.suptitle('Paper-inspired runtime models on Quantum Rings challenge data',
             fontsize=13, fontweight='bold')
fig.savefig(OUT / 'paper_runtime_ablation.png', dpi=180)
plt.close(fig)

has_seed_chart = bool(robust and all(
    len(robust.get('seeds', {}).get(split, {})) >= 3
    for split in ('matched', 'structural')))
if has_seed_chart:
    fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
    for i, split in enumerate(('matched', 'structural')):
        seeds = sorted(robust['seeds'][split], key=int)
        deltas = [robust['seeds'][split][seed]['family']['score'] -
                  robust['seeds'][split][seed]['graph']['score'] for seed in seeds]
        xs = np.arange(len(seeds)) + (i - .5) * .18
        ax.scatter(xs, deltas, label=split, s=60,
                   color=('#517dbe', '#d49557')[i])
        for x, delta in zip(xs, deltas):
            ax.text(x, delta + .0008, f'{delta:+.4f}', ha='center', fontsize=8)
    ax.axhline(0, color='#333333', linewidth=1)
    ax.set(xticks=np.arange(3), xticklabels=seeds,
           xlabel='Neural initialization seed',
           ylabel='Family FiLM + residual minus graph-only score',
           title='Family-conditioning change across neural fits')
    ax.legend()
    fig.savefig(OUT / 'paper_runtime_seed_sensitivity.png', dpi=180)
    plt.close(fig)

print(OUT / 'paper_runtime_ablation.png')
if has_seed_chart:
    print(OUT / 'paper_runtime_seed_sensitivity.png')
