"""Compare full-harness parser timing before and after the arity fast paths."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
features = json.loads((HERE / 'features.json').read_text())


def timings(path):
    rows = list(csv.DictReader(path.open(newline='')))
    return {row['filename']: float(row['parse_s']) for row in rows}


before = timings(HERE / 'training_submission_pruned.csv')
after = timings(HERE / 'training_submission_parser_optimized.csv')
names = sorted(before.keys() & after.keys())
size = np.array([features[name]['qasm_bytes']/1e6 for name in names])
old = np.array([before[name] for name in names])
new = np.array([after[name] for name in names])
top = np.argsort(old)[-12:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout='constrained')
ax = axes[0]
ax.scatter(size, old, s=20, color='#9ca4ad', alpha=.5, label='Before')
ax.scatter(size, new, s=18, color='#16877e', alpha=.55, label='After')
ax.axhline(15, color='#a44941', linestyle='--', linewidth=1.3,
           label='15 s parser cap')
ax.set(xscale='log', yscale='log', xlabel='Decoded QASM size (MB)',
       ylabel='Parser time (s)', title='All 532 training circuits')
ax.legend(frameon=False)
ax.grid(alpha=.15, which='both')

ax = axes[1]
y = np.arange(len(top))
ax.barh(y-.19, old[top], height=.36, color='#9ca4ad', label='Before')
ax.barh(y+.19, new[top], height=.36, color='#16877e', label='After')
ax.axvline(15, color='#a44941', linestyle='--', linewidth=1.3)
ax.set_yticks(y, [names[i][:8] for i in top])
ax.invert_yaxis()
ax.set(xlabel='Parser time (s)', title='Slowest 12 circuits before change')
ax.legend(frameon=False)
ax.grid(axis='x', alpha=.15)
fig.suptitle('Large-QASM parser optimization', fontsize=14, fontweight='bold')
fig.savefig(HERE / 'parser_optimization.png', dpi=180)
print(HERE / 'parser_optimization.png')
