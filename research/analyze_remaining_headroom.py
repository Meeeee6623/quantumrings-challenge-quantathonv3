"""Summarize where the selected matched-fold prediction loss remains.

    uv run --locked python research/analyze_remaining_headroom.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research'


def main():
    rows = list(csv.DictReader((OUT / 'full_union_model_oof.csv').open()))
    features = json.loads((OUT / 'features.json').read_text())
    losses = []
    by_circuit = defaultdict(float)
    by_size = defaultdict(list)
    by_duration = defaultdict(list)
    by_status = defaultdict(list)
    gt_ten = 0
    for row in rows:
        actual = float(row['actual_s'])
        predicted = float(row['matched_basis_pred_s'])
        assert actual > 0 and predicted > 0
        log_error = abs(math.log10(predicted / actual))
        loss = min(1.0, log_error / 2.0)
        circuit = row['filename']
        losses.append(loss)
        by_circuit[circuit] += loss
        by_size['>=10 MB' if features[circuit]['qasm_bytes'] >= 10_000_000 else '<10 MB'].append(loss)
        by_duration['<1 second' if actual < 1 else '>=1 second'].append(loss)
        by_status[row['status']].append(loss)
        gt_ten += log_error > 1

    total_loss = sum(losses)
    n = len(rows)
    sorted_circuits = sorted(by_circuit.values(), reverse=True)
    result = {
        'method': 'selected v7 fixed circuit-grouped matched OOF; challenge loss = min(1, |log10(pred/actual)|/2)',
        'rows': n,
        'circuits': len(by_circuit),
        'score': 1 - total_loss / n,
        'greater_than_10x_rows': gt_ten,
        'top_circuit_loss_share': {str(k): sum(sorted_circuits[:k]) / total_loss for k in (5, 10, 20, 50)},
        'oracle_gain_if_top_circuits_perfect': {str(k): sum(sorted_circuits[:k]) / n for k in (5, 10, 20, 50)},
        'size': {key: {'rows': len(values), 'score': 1 - sum(values) / len(values),
                       'loss_share': sum(values) / total_loss,
                       'oracle_gain_if_perfect': sum(values) / n}
                 for key, values in by_size.items()},
        'duration': {key: {'rows': len(values), 'score': 1 - sum(values) / len(values),
                           'loss_share': sum(values) / total_loss}
                     for key, values in by_duration.items()},
        'status': {key: {'rows': len(values), 'score': 1 - sum(values) / len(values),
                         'loss_share': sum(values) / total_loss}
                   for key, values in by_status.items()},
    }
    (OUT / 'remaining_headroom.json').write_text(json.dumps(result, indent=2) + '\n')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    ks = [5, 10, 20, 50]
    shares = [100 * result['top_circuit_loss_share'][str(k)] for k in ks]
    axes[0].bar([str(k) for k in ks], shares, color='#3854a4')
    axes[0].set(title='Loss concentrated in a few circuits',
                xlabel='Worst circuits, selected using held-out errors',
                ylabel='Share of current score loss (%)', ylim=(0, 50))
    for i, value in enumerate(shares):
        axes[0].text(i, value + 1, f'{value:.1f}%', ha='center')

    groups = ['<10 MB', '>=10 MB']
    vals = [100 * result['size'][group]['loss_share'] for group in groups]
    axes[1].bar(groups, vals, color=['#28947d', '#d18e40'])
    axes[1].set(title='Large files are a small fraction of total loss',
                xlabel='Decoded source size', ylabel='Share of current score loss (%)',
                ylim=(0, 105))
    for i, group in enumerate(groups):
        axes[1].text(i, vals[i] + 1.3,
                     f'{vals[i]:.1f}%  ({result["size"][group]["rows"]} runs)',
                     ha='center')
    fig.suptitle('Remaining headroom: matched circuit-grouped validation (v7)',
                 fontsize=12, fontweight='bold')
    fig.savefig(OUT / 'remaining_headroom.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
