"""Audit every submitted model feature against grouped runtime observations.

    uv run --locked --extra report python research/audit_selected_features.py

Writes a row-by-row decision table and contact sheets. The decisions are
exploratory: fitted impurity importance and marginal trends are correlated
feature diagnostics, not causal or held-out ablation estimates.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from train_full_union_model import ROOT, load_extended_matrix, load_training_table

ARTIFACT = ROOT / 'quantathon-harness/artifacts/runtime_model.joblib'
OUT = ROOT / 'research/feature_audit'
SETTINGS = (16, 64, 512)
COLORS = {16: '#0072B2', 64: '#D55E00', 512: '#009E73'}


def matrix_for_selected(columns, rows, primary, p_names, thresholds):
    extended, e_names = load_extended_matrix(rows, thresholds)
    all_columns = p_names + e_names
    all_values = np.column_stack((primary, extended))
    values = {name: all_values[:, i] for i, name in enumerate(all_columns)}
    values.update({f'setting_{setting}': (thresholds == setting).astype(float)
                   for setting in SETTINGS})
    missing = set(columns) - set(values)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    return np.column_stack([values[name] for name in columns])


def importances(artifact, thresholds):
    names = artifact['columns']
    roles = {'global_runtime': (artifact['global_columns'],
                                artifact['global_estimator'])}
    for setting in SETTINGS:
        roles[f'runtime_{setting}'] = (
            artifact['specialist_columns_by_threshold'][setting],
            artifact['threshold_specialists'][setting])
        roles[f'timeout_{setting}'] = (
            artifact['classifier_columns_by_threshold'][setting],
            artifact['timeout_classifiers'][setting])
    weights = {setting: float(np.mean(thresholds == setting)) for setting in SETTINGS}
    by_name = {name: {} for name in names}
    for role, (columns, estimator) in roles.items():
        if len(columns) != len(estimator.feature_importances_):
            raise ValueError(f'Invalid fitted importance shape for {role}')
        current = dict(zip(columns, estimator.feature_importances_))
        for name in names:
            by_name[name][role] = float(current.get(name, 0.0))
    for name, row in by_name.items():
        row['runtime_importance'] = (
            .5 * row['global_runtime'] +
            .5 * sum(weights[t] * row[f'runtime_{t}'] for t in SETTINGS))
        row['timeout_importance'] = sum(
            weights[t] * row[f'timeout_{t}'] for t in SETTINGS)
    return by_name


def binned_curve(x, y, bins=7):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10 or len(np.unique(x)) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    unique = np.unique(x)
    if len(unique) <= bins:
        labels = np.searchsorted(unique, x)
        number_of_bins = len(unique)
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        labels = np.digitize(x, edges[1:-1], right=True)
        number_of_bins = len(edges) - 1
    stats = [(np.median(x[labels == i]), np.median(y[labels == i]),
              np.quantile(y[labels == i], .25),
              np.quantile(y[labels == i], .75))
             for i in range(number_of_bins) if np.count_nonzero(labels == i) >= 5]
    if not stats:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return tuple(np.asarray(v) for v in zip(*stats))


def view(name):
    if name.startswith('extended__'):
        return 'secondary scanner'
    if name.startswith('chi_walk_'):
        return 'angle-aware chi walk'
    if name.startswith('setting_') or name in ('threshold', 'log_threshold'):
        return 'simulator setting'
    if name.startswith('log_'):
        return 'log1p primary transform'
    return 'primary parser'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(ARTIFACT)
    if artifact['artifact_version'] != 'pruned_union_threshold_experts_v7':
        raise SystemExit('This prescreen audit requires the v7 artifact; '
                         'run research/train_full_union_model.py first. '
                         'Existing audit outputs are safe to read with v8.')
    rows, primary, p_names, y, timeout, thresholds, names = load_training_table()
    columns = artifact['columns']
    X = matrix_for_selected(columns, rows, primary, p_names, thresholds)
    imp = importances(artifact, thresholds)
    rank = np.column_stack([rankdata(np.nan_to_num(X[:, i], nan=-1e9))
                            for i in range(X.shape[1])])
    corr = np.corrcoef(rank, rowvar=False)
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0)
    by_name = {name: i for i, name in enumerate(columns)}
    catalog = []
    for j, name in enumerate(columns):
        x = X[:, j]
        rho = {}
        spread = {}
        curves = {}
        for setting in SETTINGS:
            mask = thresholds == setting
            xt, yt = x[mask], y[mask]
            valid = np.isfinite(xt)
            if np.count_nonzero(valid) > 5 and len(np.unique(xt[valid])) > 1:
                rho[setting] = float(spearmanr(xt[valid], yt[valid]).statistic)
                curve = binned_curve(xt, yt)
                spread[setting] = (float(np.ptp(curve[1])) if len(curve[1]) > 1
                                   else 0.0)
                curves[setting] = curve
            else:
                rho[setting] = None
                spread[setting] = 0.0
                curves[setting] = (np.array([]),) * 4
        rivals = [i for i in np.flatnonzero(np.abs(corr[j]) >= .995)
                  if imp[columns[i]]['runtime_importance'] +
                     imp[columns[i]]['timeout_importance'] >
                     imp[name]['runtime_importance'] + imp[name]['timeout_importance']]
        rival = max(rivals, key=lambda i: (imp[columns[i]]['runtime_importance']+
                                         imp[columns[i]]['timeout_importance'])) if rivals else None
        runtime_imp = imp[name]['runtime_importance']
        timeout_imp = imp[name]['timeout_importance']
        max_rho = max((abs(value) for value in rho.values() if value is not None),
                      default=0.0)
        max_spread = max(spread.values())
        if name in ('threshold', 'log_threshold', 'extended__threshold'):
            decision, reason = ('drop', 'Replaced by categorical one-hot setting; '
                                'the ordinal encoding adds an unjustified distance.')
        elif name.startswith('setting_'):
            decision, reason = 'keep', 'Categorical simulator setting for the global runtime model.'
        elif rival is not None and runtime_imp + timeout_imp < .002:
            decision, reason = ('drop', f'Near-perfect rank duplicate of {columns[rival]} '
                                f'(|rho|={abs(corr[j, rival]):.3f}) with lower fitted importance.')
        elif runtime_imp == 0 and timeout_imp == 0:
            decision, reason = 'drop', 'Zero fitted importance in every runtime and timeout component.'
        elif runtime_imp < .00015 and timeout_imp < .00015 and max_rho < .12 and max_spread < .5:
            decision, reason = ('drop', 'Tiny fitted importance and no clear within-setting '
                                'runtime trend in the grouped curves.')
        elif runtime_imp >= .0005 or timeout_imp >= .0005:
            decision, reason = ('keep', 'Material fitted runtime or timeout importance; '
                                'grouped curve supplies marginal context.')
        elif max_rho >= .12 or max_spread >= .5:
            decision, reason = ('keep', 'Low fitted importance but a visible within-setting '
                                'runtime trend; retain pending grouped ablation.')
        else:
            decision, reason = ('keep', 'Small nonzero fitted importance; retain pending '
                                'grouped ablation because interactions can be marginally flat.')
        role_flags = {role: int(imp[name][role] > 0)
                      for role in ('global_runtime','runtime_16','runtime_64',
                                   'runtime_512','timeout_16','timeout_64','timeout_512')}
        catalog.append({
            'ordinal': j + 1, 'feature': name, 'view': view(name),
            'decision': decision, 'reason': reason,
            'runtime_importance': runtime_imp,
            'timeout_importance': timeout_imp,
            'rho_16': rho[16], 'rho_64': rho[64], 'rho_512': rho[512],
            'binned_log10_spread_16': spread[16],
            'binned_log10_spread_64': spread[64],
            'binned_log10_spread_512': spread[512],
            'redundant_with': columns[rival] if rival is not None else '',
            'plot_sheet': f'grouped_curves_{j // 16 + 1:02d}.png',
            **role_flags,
            '_curves': curves,
        })
    fields = [key for key in catalog[0] if not key.startswith('_')]
    with (OUT / 'feature_decisions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in catalog)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.boxplot([y[thresholds == setting] for setting in SETTINGS],
                 tick_labels=[str(setting) for setting in SETTINGS],
                 showfliers=False)
    axis.set(xlabel='Simulator setting (category)',
             ylabel='Observed log10(runtime seconds)',
             title='Observed runtime by simulator setting')
    axis.grid(axis='y', alpha=.2)
    fig.savefig(OUT / 'setting_vs_runtime.png', dpi=160)
    plt.close(fig)
    for first in range(0, len(catalog), 16):
        fig, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
        for axis, row in zip(axes.flat, catalog[first:first + 16]):
            for setting in SETTINGS:
                cx, cy, lower, upper = row['_curves'][setting]
                if not len(cx):
                    continue
                axis.plot(cx, cy, 'o-', markersize=2.8, linewidth=1.3,
                          color=COLORS[setting], label=str(setting))
                axis.fill_between(cx, lower, upper, alpha=.09,
                                  color=COLORS[setting])
            j = row['ordinal'] - 1
            valid_x = X[np.isfinite(X[:, j]), j]
            if len(valid_x) and np.nanmax(np.abs(valid_x)) > 1000 and np.ptp(valid_x) > 1000:
                axis.set_xscale('symlog', linthresh=1)
            axis.set_title(f"{row['ordinal']}. {row['feature']} [{row['decision']}]\n"
                           f"runtime={row['runtime_importance']:.4g}, "
                           f"timeout={row['timeout_importance']:.4g}")
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=7)
            axis.set_ylabel('actual log10(seconds)')
            if first == 0 and row['ordinal'] == 1:
                axis.legend(title='setting', fontsize=7)
        for axis in axes.flat[len(catalog[first:first + 16]):]:
            axis.set_visible(False)
        page = first // 16 + 1
        fig.suptitle(f'Observed runtime by feature, grouped by categorical simulator setting — '
                     f'features {first + 1}–{min(first + 16, len(catalog))}', fontsize=14)
        fig.savefig(OUT / f'grouped_curves_{page:02d}.png', dpi=140)
        plt.close(fig)
    counts = {decision: sum(row['decision'] == decision for row in catalog)
              for decision in ('keep', 'drop')}
    (OUT / 'audit_summary.json').write_text(json.dumps({
        'artifact_version': artifact['artifact_version'], 'features': len(catalog),
        'decisions': counts, 'circuits': len(set(names)), 'rows': len(rows),
        'settings': list(SETTINGS),
        'note': 'Within-setting binned medians and IQR are descriptive; '
                'decisions require grouped ablation before deployment.'}, indent=2) + '\n')
    with (OUT / 'FEATURE_BY_FEATURE_AUDIT.md').open('w') as handle:
        handle.write('# Feature-by-feature audit\n\n')
        handle.write(f"Fitted artifact: `{artifact['artifact_version']}`. "
                     f"{len(catalog)} inputs; {counts['keep']} preliminary keeps and "
                     f"{counts['drop']} preliminary drops. The target is observed "
                     'log10 runtime in seconds (timeouts capped by the challenge). '
                     'Every curve bins a feature within each of the three simulator '
                     'settings and shows the median with interquartile shading. '
                     'The simulator setting is treated as categorical in the plots. '
                     'Impurity importances are weighted across the global and setting '
                     'experts; correlated inputs divide importance. A flat marginal '
                     'plot does not rule out an interaction. These are proposed '
                     'decisions until grouped ablation is validated.\n\n')
        handle.write('Runtime importance weights: 50% global and 50% setting '
                     'experts, weighted by setting frequency. Timeout importance '
                     'averages the three timeout classifiers. Spearman correlations '
                     'are separately computed inside each setting.\n\n')
        handle.write('![Observed runtime by categorical simulator setting]'
                     '(setting_vs_runtime.png)\n\n')
        for page in range(1, (len(catalog) + 15) // 16 + 1):
            handle.write(f'![Grouped curves page {page}](grouped_curves_{page:02d}.png)\n\n')
        handle.write('| # | Feature | Decision | Runtime imp. | Timeout imp. | '
                     'ρ 16 / 64 / 512 | Reason |\n')
        handle.write('|---:|---|---|---:|---:|---|---|\n')
        for row in catalog:
            rhos = '/'.join('—' if row[f'rho_{t}'] is None else
                            f"{row[f'rho_{t}']:.2f}" for t in SETTINGS)
            handle.write(f"| {row['ordinal']} | `{row['feature']}` | "
                         f"[{row['decision']}](./{row['plot_sheet']}) | "
                         f"{row['runtime_importance']:.5f} | "
                         f"{row['timeout_importance']:.5f} | {rhos} | "
                         f"{row['reason']} |\n")
    print(json.dumps({'features': len(catalog), 'decisions': counts,
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
