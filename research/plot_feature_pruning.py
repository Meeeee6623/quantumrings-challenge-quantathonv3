"""Rebuild the feature-pruning validation figure from its JSON report."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'feature_pruning_validation.json').read_text())
labels = ['Full 480', 'Exact dedup', '120 / 200 / 80', '80 / 140 / 60']
keys = ['baseline', 'deduplicated', 'moderate', 'aggressive']
colors = ['#5b6471', '#91a4b7', '#127c78', '#b36b47']
matched, stress, matched_miss, stress_miss = [], [], [], []
for key in keys:
    data = report['baseline'] if key == 'baseline' else report['candidates'][key]['splits']
    matched.append(data['matched']['score'])
    stress.append(data['structural']['score'])
    matched_miss.append(data['matched']['tenfold_misses'])
    stress_miss.append(data['structural']['tenfold_misses'])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout='constrained')
x = np.arange(len(keys))
axes[0].bar(x - .18, 100*(np.array(matched)-matched[0]), .36,
            color=colors, label='Matched')
axes[0].bar(x + .18, 100*(np.array(stress)-stress[0]), .36,
            color=colors, alpha=.48, label='Structural stress')
axes[0].axhline(0, color='#222', linewidth=.8)
axes[0].set_ylim(-.25, .65)
axes[0].set_ylabel('Score change vs full (percentage points)')
axes[0].set_title('Circuit-grouped validation')
axes[0].legend(frameon=False, loc='lower left')
axes[1].bar(x - .18, np.array(matched_miss)-matched_miss[0], .36, color=colors)
axes[1].bar(x + .18, np.array(stress_miss)-stress_miss[0], .36,
            color=colors, alpha=.48)
axes[1].axhline(0, color='#222', linewidth=.8)
axes[1].set_ylim(-9, 3)
axes[1].set_ylabel('Change in >10× error rows')
axes[1].set_title('Large prediction errors')
widths = np.array([[243,480,480], [241,456,456], [120,200,80], [80,140,60]])
for j, label in enumerate(['Global', 'Runtime experts', 'Timeout classifiers']):
    axes[2].plot(x, widths[:,j], marker='o', linewidth=2, label=label)
axes[2].set_ylim(0,520)
axes[2].set_ylabel('Input columns per model')
axes[2].set_title('Model width')
axes[2].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.set_xticks(x, labels, rotation=24, ha='right')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('Quantum Rings feature pruning: 1,497 labeled rows / 532 circuits',
             fontsize=13, fontweight='bold')
fig.savefig(HERE / 'feature_pruning.png', dpi=180)
print(HERE / 'feature_pruning.png')
