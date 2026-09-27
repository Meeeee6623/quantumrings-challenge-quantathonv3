"""Reconcile the one-by-one v7 audit with the categorical v8 fitted schema.

    uv run --locked --extra report python research/write_feature_decisions.py
"""
from __future__ import annotations

import csv
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_selected_features import (
    ARTIFACT, COLORS, OUT, SETTINGS, binned_curve, importances,
    matrix_for_selected,
)
from train_full_union_model import load_training_table

ROOT = OUT.parent.parent
CATALOG = ROOT / 'research/final_feature_catalog.csv'
PROBE = ROOT / 'research/categorical_setting_probe.json'
PRUNE = ROOT / 'research/feature_audit_prune_probe.json'
FINAL = ROOT / 'research/categorical_model_validation.json'


def main():
    audit = list(csv.DictReader((OUT / 'feature_decisions.csv').open()))
    selected = {row['feature'] for row in csv.DictReader(CATALOG.open())}
    artifact = joblib.load(ARTIFACT)
    training_rows, primary, p_names, y, _, thresholds, _ = load_training_table()
    final_importance = importances(artifact, thresholds)
    assert set(final_importance) == selected
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, key, title in (
        (axes[0], 'runtime_importance', 'Runtime model'),
        (axes[1], 'timeout_importance', 'Timeout classifiers'),
    ):
        top = sorted(selected, key=lambda name: final_importance[name][key],
                     reverse=True)[:15]
        axis.barh(top[::-1], [final_importance[name][key] for name in top[::-1]],
                  color='#0072B2' if key == 'runtime_importance' else '#D55E00')
        axis.set_title(f'Top 15 fitted importances: {title}')
        axis.set_xlabel('Weighted impurity importance')
        axis.grid(axis='x', alpha=.2)
    fig.savefig(OUT / 'top_feature_importance.png', dpi=150)
    plt.close(fig)
    top_runtime = sorted((name for name in selected if not name.startswith('setting_')),
                         key=lambda name:
                         final_importance[name]['runtime_importance'],
                         reverse=True)[:12]
    X = matrix_for_selected(top_runtime, training_rows, primary, p_names, thresholds)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    for j, (axis, name) in enumerate(zip(axes.flat, top_runtime)):
        for setting in SETTINGS:
            mask = thresholds == setting
            cx, cy, lower, upper = binned_curve(X[mask, j], y[mask])
            if len(cx):
                axis.plot(cx, cy, 'o-', color=COLORS[setting],
                          markersize=3, linewidth=1.4, label=str(setting))
                axis.fill_between(cx, lower, upper, color=COLORS[setting], alpha=.1)
        if np.nanmax(np.abs(X[:, j])) > 1000 and np.ptp(X[:, j]) > 1000:
            axis.set_xscale('symlog', linthresh=1)
        axis.set_title(name)
        axis.set_ylabel('actual log10(seconds)')
        axis.grid(alpha=.2)
    axes.flat[0].legend(title='setting', fontsize=8)
    fig.suptitle('Top fitted runtime features versus observed runtime, grouped by setting',
                 fontsize=14)
    fig.savefig(OUT / 'top_runtime_grouped_curves.png', dpi=140)
    plt.close(fig)
    rows = []
    for old in audit:
        name = old['feature']
        kept = name in selected
        if kept:
            reason = ('Retained after categorical encoding and grouped ablation; '
                      'has nonzero final fitted importance.')
        elif old['decision'] == 'drop':
            reason = old['reason']
        else:
            reason = ('Passed the initial audit but was not selected in any '
                      'final runtime or timeout estimator.')
        rows.append({
            'ordinal': len(rows) + 1, 'feature': name,
            'view': old['view'], 'final_decision': 'keep' if kept else 'drop',
            'reason': reason,
            'v7_runtime_importance': old['runtime_importance'],
            'v7_timeout_importance': old['timeout_importance'],
            'v8_runtime_importance':
                final_importance[name]['runtime_importance'] if kept else 0.0,
            'v8_timeout_importance':
                final_importance[name]['timeout_importance'] if kept else 0.0,
            'rho_16': old['rho_16'], 'rho_64': old['rho_64'],
            'rho_512': old['rho_512'],
            'plot': old['plot_sheet'],
            'redundant_with': old['redundant_with'],
        })
    for setting in (16, 64, 512):
        name = f'setting_{setting}'
        assert name in selected
        rows.append({
            'ordinal': len(rows) + 1, 'feature': name,
            'view': 'categorical simulator setting',
            'final_decision': 'keep',
            'reason': ('One-hot category used by the global model; specialists '
                       'and timeout classifiers are selected by the category.'),
            'v7_runtime_importance': '', 'v7_timeout_importance': '',
            'v8_runtime_importance': final_importance[name]['runtime_importance'],
            'v8_timeout_importance': final_importance[name]['timeout_importance'],
            'rho_16': '', 'rho_64': '', 'rho_512': '',
            'plot': 'setting_vs_runtime.png', 'redundant_with': '',
        })
    assert len(rows) == 328
    assert sum(row['final_decision'] == 'keep' for row in rows) == len(selected)
    fields = list(rows[0])
    with (OUT / 'final_decisions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cat = json.loads(PROBE.read_text())
    prune = json.loads(PRUNE.read_text())
    final = json.loads(FINAL.read_text())
    with (OUT / 'FINAL_FEATURE_DECISIONS.md').open('w') as handle:
        handle.write('# Final feature decisions and grouped runtime plots\n\n')
        handle.write('The release model treats the simulator setting as the '
                     'three categories **16, 64, and 512**. It uses three '
                     'one-hot fields in the global model; the specialists and '
                     'timeout classifiers are routed by setting. Numeric '
                     '`threshold` and `log_threshold` were removed. '
                     'The table below makes one decision for every input '
                     'referenced by the previous v7 artifact, plus the three '
                     'new setting indicators: **241 keep, 87 drop**. All 241 '
                     'keeps are in the fitted v8 artifact. The prior v7 model '
                     'had 325 selected inputs; its plot gallery covers those '
                     '325 one by one. The category plot covers the three new '
                     'indicators. Other parser outputs never selected by v7 '
                     'are outside this fitted-input audit.\n\n')
        handle.write('Of the 87 final drops, **78** were near-perfect rank '
                     'duplicates with a stronger retained counterpart, '
                     '**4** had tiny fitted importance and weak grouped '
                     'trends, **2** were ordinal encodings replaced by '
                     'categories, and **3** passed the prescreen but were '
                     'not selected in the final fit.\n\n')
        handle.write('![Runtime grouped by categorical setting]'
                     '(setting_vs_runtime.png)\n\n')
        handle.write('![Most important fitted features]'
                     '(top_feature_importance.png)\n\n')
        handle.write('![Grouped observed-runtime curves for the 12 most '
                     'important runtime inputs](top_runtime_grouped_curves.png)\n\n')
        handle.write('| Circuit-grouped score ↑ | Previous v7 | Full categorical | '
                     'Categorical + audit pruning |\n|---|---:|---:|---:|\n')
        for split in ('matched', 'structural'):
            base = cat['splits'][split]['current_ordinal_v7']['score']
            categorical = cat['splits'][split]['categorical']['score']
            pruned = prune['splits'][split]['categorical_audit_pruned']['score']
            assert abs(pruned - final['splits'][split]['categorical_v8']['score']) < 1e-12
            handle.write(f'| {split} | {base:.6f} | {categorical:.6f} | '
                         f'**{pruned:.6f}** |\n')
        handle.write('\nThe first full-categorical comparison changed score '
                     'by −0.00011 matched and −0.00111 structural relative '
                     'to v7, within paired uncertainty. Removing the audit '
                     'candidates then improved both fixed splits. The '
                     'pruned-vs-full paired score differences were '
                     f"{prune['splits']['matched']['paired']['delta_merged_minus_current']:+.6f} "
                     'matched (95% circuit bootstrap interval '
                     f"{prune['splits']['matched']['paired']['delta_95']}) and "
                     f"{prune['splits']['structural']['paired']['delta_merged_minus_current']:+.6f} "
                     'structural (cluster bootstrap interval '
                     f"{prune['splits']['structural']['paired']['delta_95']}). "
                     'The screening decisions used all released labels before '
                     'this ablation, so these validation gains can be optimistic; '
                     'they are not hidden-holdout results.\n\n')
        handle.write('**How to read the table.** Fitted impurity importance is '
                     'weighted 50/50 between global and setting runtime '
                     'regressors; timeout importance averages setting-specific '
                     'classifiers. The ρ values are Spearman correlations '
                     'between feature and observed log10 seconds **inside** '
                     'each setting. Each linked plot shows within-setting '
                     'binned median runtime with interquartile shading. '
                     'A feature with a flat marginal curve can still matter '
                     'through interactions; near-perfectly ranked raw/log '
                     'pairs divide tree importance. The v7 importances and '
                     'curves drove the drop proposals, while v8 importances '
                     'describe the final kept model.\n\n')
        handle.write('| # | Feature | Decision | v7 runtime / timeout imp. | '
                     'v8 runtime / timeout imp. | ρ 16 / 64 / 512 | Reason |\n')
        handle.write('|---:|---|---|---:|---:|---|---|\n')
        for row in rows:
            def fmt(value):
                return '—' if value == '' else f'{float(value):.4g}'
            imp_old = (fmt(row['v7_runtime_importance']) + ' / ' +
                       fmt(row['v7_timeout_importance']))
            imp_new = (fmt(row['v8_runtime_importance']) + ' / ' +
                       fmt(row['v8_timeout_importance']))
            rho = ' / '.join(fmt(row[f'rho_{t}']) for t in (16, 64, 512))
            handle.write(f"| {row['ordinal']} | `{row['feature']}` | "
                         f"[{row['final_decision']}](./{row['plot']}) | "
                         f'{imp_old} | {imp_new} | {rho} | {row["reason"]} |\n')
    print('Wrote 328 final feature decisions; kept', len(selected))


if __name__ == '__main__':
    main()
