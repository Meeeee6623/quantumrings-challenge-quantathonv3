"""Plot grouped-fold template-analogue weight comparisons."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / 'template_weight_probe.json').read_text())
groups = [('Fixed matched', report['fixed']['matched'])]
groups += [(f'Alternate {seed}', result)
           for seed, result in sorted(report['alternate_seeds'].items())]
labels = [name for name, _ in groups]
variants = [('0.75', '75% analogue', '#7896a6'),
            ('1.0', '100% analogue', '#c27d48'),
            ('adaptive', '50% with 2 refs; 100% with 3+', '#117c78')]
x = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
for j, (key, label, color) in enumerate(variants):
    delta = [10000 * (result['weights'][key]['score']
                      - result['weights']['0.5']['score'])
             for _, result in groups]
    ax.bar(x + (j-1)*.24, delta, width=.22, label=label, color=color)
ax.axhline(0, color='#222', linewidth=.9)
ax.set_xticks(x, labels)
ax.set_ylabel('Challenge-score gain vs current blend (basis points)')
ax.set_title('Template weighting on circuit-grouped folds')
ax.legend(frameon=False, fontsize=8)
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
fig.savefig(HERE / 'template_weight.png', dpi=180)
print(HERE / 'template_weight.png')
