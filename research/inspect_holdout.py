"""Inspect new circuits against training geometry and grouped-OOF errors.

    uv run --locked python research/inspect_holdout.py \
      --circuits path/to/holdout --out-dir inspection

This never uses holdout labels and does not change the submitted model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'quantathon-harness'))
from model import RuntimeModel  # noqa: E402
from run import read_qasm  # noqa: E402
from template_analogues import signature  # noqa: E402

GATE_FIELDS = ('n_qubits', 'ops', 'two_q', 'multi_q', 'effective_ops', 'depth',
               'resets', 'chi_upper_mid', 'qasm_bytes', 'gate_h', 'gate_x',
               'gate_rx', 'gate_ry', 'gate_rz', 'gate_u', 'gate_u3', 'gate_cx',
               'gate_cz', 'gate_cp', 'gate_swap', 'gate_rzz', 'gate_ccx',
               'measurements')
SHAPE_FIELDS = ('n_qubits', 'ops', 'two_q', 'multi_q', 'effective_ops', 'depth',
                'resets', 'chi_upper_mid', 'qasm_bytes', 'unique_pairs',
                'pair_reuse', 'graph_cutwidth_rcm', 'graph_degree_entropy',
                'randomness_proxy', 'chi_est_log2_peak',
                'chi_capacity_fraction', 'two_q_ratio', 'nonclifford_ratio',
                'liveness_peak_window')
FRACTIONS = {'randomness_proxy', 'chi_capacity_fraction', 'two_q_ratio',
             'nonclifford_ratio', 'graph_degree_entropy'}
GEOMETRY_FIELDS = ('n_qubits', 'active_qubits', 'ops', 'one_q', 'two_q',
                   'multi_q', 'effective_ops', 'depth', 'two_depth',
                   'unique_pairs', 'pair_reuse', 'graph_cutwidth_rcm',
                   'graph_degree_entropy', 'chi_upper_mid',
                   'chi_est_log2_peak', 'randomness_proxy', 'resets',
                   'measurements', 'qasm_bytes', 'huge_fast_path')


def circuit_files(directory: Path, name: str | None = None):
    if not directory.is_dir():
        raise ValueError(f'Circuit directory does not exist: {directory}')
    files = sorted(p for p in directory.rglob('*') if p.is_file() and
                   (p.name.endswith('.qasm') or p.name.endswith('.qasm.zst')))
    if name:
        files = [p for p in files if qasm_name(p) == name]
    if not files:
        raise ValueError(f'No matching QASM files under {directory}')
    names = [qasm_name(p) for p in files]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate QASM basenames would collide in the harness')
    return files


def qasm_name(path: Path):
    return path.name[:-4] if path.name.endswith('.qasm.zst') else path.name


def raw_matrix(features: list[dict], fields: tuple[str, ...]):
    matrix = np.asarray([[float(feat.get(field, 0) or 0) for field in fields]
                         for feat in features], dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    for j, field in enumerate(fields):
        if field not in FRACTIONS:
            matrix[:, j] = np.log1p(np.maximum(matrix[:, j], 0))
    return matrix


def build_reference(training: list[dict], fields: tuple[str, ...]):
    raw = raw_matrix(training, fields)
    center = np.median(raw, axis=0)
    scale = np.maximum(np.std(raw, axis=0), 0.1)
    standardized = (raw - center) / scale
    pairwise = cdist(standardized, standardized) / math.sqrt(len(fields))
    np.fill_diagonal(pairwise, np.inf)
    leave_one_out = pairwise.min(axis=1)
    return {'fields': fields, 'center': center, 'scale': scale,
            'standardized': standardized,
            'training_nearest_p95': float(np.quantile(leave_one_out, 0.95))}


def distances_to_training(reference: dict, target: dict):
    query = ((raw_matrix([target], reference['fields']) - reference['center'])
             / reference['scale'])
    return (cdist(query, reference['standardized'])[0]
            / math.sqrt(len(reference['fields'])))


def feature_bounds(training: list[dict]):
    return {field: np.quantile(
        [float(item.get(field, 0) or 0) for item in training], [0.01, 0.99])
        for field in GEOMETRY_FIELDS}


def unusual_fields(bounds: dict, target: dict):
    result = []
    for field in GEOMETRY_FIELDS:
        value = float(target.get(field, 0) or 0)
        if not np.isfinite(value):
            continue
        low, high = bounds[field]
        if value < low or value > high:
            result.append({'feature': field, 'value': value,
                           'training_p01': float(low),
                           'training_p99': float(high),
                           'direction': 'below' if value < low else 'above'})
    return result


def training_outcome(oof: dict, name: str, threshold: int):
    row = oof.get((name, threshold))
    if not row:
        return None
    actual = float(row['actual_s'])
    predicted = float(row.get('matched_pred_s', row.get('matched_basis_pred_s')))
    factor = max(actual / predicted, predicted / actual)
    return {'actual_s': actual, 'oof_pred_s': predicted,
            'oof_factor_error': factor,
            'oof_score': max(0.0, 1 - math.log10(factor) / 2),
            'status': row['status'],
            'label': 'missed' if factor >= 3 else 'close' if factor <= 1.25 else 'middle'}


def inspect_one(path, model, training_names, training, references, bounds,
                oof, thresholds, top_k):
    started = time.perf_counter()
    qasm = read_qasm(path)
    features = model.featurize(qasm)
    parse_s = time.perf_counter() - started
    name = qasm_name(path)
    geometric = {field: features.get(field, 0) for field in GEOMETRY_FIELDS}
    views = {}
    for view_name in ('gate_mix', 'shape'):
        reference = references[view_name]
        distances = distances_to_training(reference, features)
        if name in training_names:
            distances[training_names.index(name)] = np.inf
        p95 = reference['training_nearest_p95']
        ordered = np.argsort(distances)
        nearest = []
        for i in ordered[:max(top_k, 20)]:
            training_name = training_names[int(i)]
            nearest.append({'filename': training_name,
                            'distance': float(distances[i]),
                            'geometry': {key: training[i].get(key, 0)
                                         for key in ('n_qubits', 'ops', 'two_q',
                                                     'multi_q', 'chi_upper_mid')},
                            'oof': {str(t): training_outcome(oof, training_name, t)
                                    for t in thresholds}})
        views[view_name] = {
            'nearest_distance': float(distances[ordered[0]]),
            'training_nearest_p95': p95,
            'outside_training_p95': bool(distances[ordered[0]] > p95),
            'nearest': nearest[:top_k],
            'nearby_missed': [n for n in nearest if any(
                x and x['label'] == 'missed' for x in n['oof'].values())][:top_k],
            'nearby_close': [n for n in nearest if any(
                x and x['label'] == 'close' for x in n['oof'].values())][:top_k],
        }
    exact = [training_names[i] for i, value in enumerate(training)
             if training_names[i] != name and signature(value) == signature(features)]
    predictions = {str(t): float(model.predict(features, t)) for t in thresholds}
    gate_counts = sorted(((key, int(value)) for key, value in features.items()
                          if key.startswith('gate_') and value),
                         key=lambda item: item[1], reverse=True)
    return {'filename': name, 'source': str(path), 'parse_s': parse_s,
            'comparison_excluded_same_filename': name in training_names,
            'predictions_s': predictions, 'geometry': geometric,
            'top_gates': gate_counts[:12], 'unusual_fields': unusual_fields(bounds, features),
            'exact_signature_training_circuits': exact, 'views': views}


def plot_overview(out_dir, records):
    if not records:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    ops = [max(1, r['geometry']['ops']) for r in records]
    shape = [r['views']['shape']['nearest_distance'] for r in records]
    gate = [r['views']['gate_mix']['nearest_distance'] for r in records]
    shape_limit = records[0]['views']['shape']['training_nearest_p95']
    gate_limit = records[0]['views']['gate_mix']['training_nearest_p95']
    for ax, distances, limit, title in (
            (axes[0], shape, shape_limit, 'Shape distance to nearest training circuit'),
            (axes[1], gate, gate_limit, 'Gate-mix distance to nearest training circuit')):
        colors = ['#d37131' if value > limit else '#3b5ba9'
                  for value in distances]
        ax.scatter(ops, distances, c=colors, alpha=0.72, s=28)
        ax.axhline(limit, color='#9b3b3b', linestyle='--', linewidth=1,
                   label='Training nearest-neighbor p95')
        ax.set(xscale='log', xlabel='Parsed operation count',
               ylabel='Standardized distance', title=title)
        ax.legend(loc='upper left', fontsize=8)
        top = np.argsort(distances)[-min(5, len(records)):]
        for i in top:
            ax.annotate(records[int(i)]['filename'].replace('.qasm', ''),
                        (ops[int(i)], distances[int(i)]), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
    fig.suptitle('Holdout geometry relative to training circuits', fontweight='bold')
    fig.savefig(out_dir / 'geometry_overview.png', dpi=170)
    plt.close(fig)


def write_outputs(out_dir, records, failures, thresholds, plot=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'inspection.json').write_text(json.dumps({
        'method': 'QASM-derived geometry vs training features; training errors are circuit-grouped matched OOF',
        'circuits': records, 'failures': failures}, indent=2) + '\n')
    with (out_dir / 'inspection.csv').open('w', newline='') as file:
        fields = ['filename', 'threshold', 'pred_duration_s', 'parse_s',
                  'n_qubits', 'ops', 'two_q', 'chi_upper_mid',
                  'gate_nearest_distance', 'gate_outside_training_p95',
                  'shape_nearest_distance', 'shape_outside_training_p95',
                  'unusual_field_count', 'exact_signature_count']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for threshold in thresholds:
                writer.writerow({
                    'filename': record['filename'], 'threshold': threshold,
                    'pred_duration_s': record['predictions_s'][str(threshold)],
                    'parse_s': round(record['parse_s'], 4),
                    **{key: record['geometry'][key]
                       for key in ('n_qubits', 'ops', 'two_q', 'chi_upper_mid')},
                    'gate_nearest_distance': record['views']['gate_mix']['nearest_distance'],
                    'gate_outside_training_p95': record['views']['gate_mix']['outside_training_p95'],
                    'shape_nearest_distance': record['views']['shape']['nearest_distance'],
                    'shape_outside_training_p95': record['views']['shape']['outside_training_p95'],
                    'unusual_field_count': len(record['unusual_fields']),
                    'exact_signature_count': len(record['exact_signature_training_circuits']),
                })
    ordered = sorted(records, key=lambda r: (
        int(r['views']['gate_mix']['outside_training_p95']) +
        int(r['views']['shape']['outside_training_p95']),
        len(r['unusual_fields']), r['views']['shape']['nearest_distance']),
        reverse=True)
    if plot:
        plot_overview(out_dir, records)
    lines = ['# Circuit inspection', '',
             f'Inspected **{len(records)}** circuits; **{len(failures)}** read/parse failures.',
             'Distance flags mean farther from training circuits than 95% of training circuits are from their own nearest neighbor. They are heuristic, not runtime-error probabilities.',
             'Training-label outcomes shown for neighbors are circuit-grouped out-of-fold diagnostics; holdout labels are never used.',
             '']
    if plot and records:
        lines += ['![Geometry overview](geometry_overview.png)', '']
    lines += ['| Circuit | Qubits | Ops | Gate OOD | Shape OOD | Unusual fields | Predictions (s) |',
             '|---|---:|---:|:---:|:---:|---:|---|']
    for record in ordered[:50]:
        geom = record['geometry']; views = record['views']
        prediction = ', '.join(f'{t}: {record["predictions_s"][str(t)]:.3g}'
                               for t in thresholds)
        lines.append(f'| {record["filename"]} | {geom["n_qubits"]} | {geom["ops"]} | '
                     f'{"yes" if views["gate_mix"]["outside_training_p95"] else ""} | '
                     f'{"yes" if views["shape"]["outside_training_p95"] else ""} | '
                     f'{len(record["unusual_fields"])} | {prediction} |')
    lines += ['', '## Highest-priority circuits', '']
    for record in ordered[:min(12, len(ordered))]:
        lines += [f'### {record["filename"]}', '',
                  f'Geometry: {json.dumps(record["geometry"], sort_keys=True)}', '',
                  'Unusual fields: ' + (', '.join(
                      f'{x["feature"]}={x["value"]:g} ({x["direction"]} training p01–p99)'
                      for x in record['unusual_fields']) or 'none') + '.', '']
        for view_name in ('gate_mix', 'shape'):
            view = record['views'][view_name]
            lines.append(f'{view_name} nearest training circuits:')
            for neighbor in view['nearest']:
                outcomes = ', '.join(
                    f'{t}: {x["oof_factor_error"]:.2g}× {x["label"]}'
                    for t, x in neighbor['oof'].items() if x)
                lines.append(f'- {neighbor["filename"]} (distance {neighbor["distance"]:.3f}; '
                             f'grouped-OOF {outcomes or "unlabeled"})')
            lines.append('')
    if failures:
        lines += ['## Failures', '']
        lines += [f'- {item["filename"]}: {item["error"]}' for item in failures]
    (out_dir / 'inspection.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuits', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'inspection')
    parser.add_argument('--thresholds', default='16,64,512')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--name', help='Inspect only this QASM basename')
    parser.add_argument('--max-circuits', type=int,
                        help='Limit circuits for a quick inspection smoke test')
    parser.add_argument('--plot', action='store_true',
                        help='Also render a geometry-distance overview PNG (requires report extra)')
    args = parser.parse_args()
    if args.top_k < 1 or (args.max_circuits is not None and args.max_circuits < 1):
        parser.error('--top-k and --max-circuits must be positive')
    thresholds = [int(item) for item in args.thresholds.split(',')]
    files = circuit_files(args.circuits, args.name)
    if args.max_circuits:
        files = files[:args.max_circuits]
    training_map = json.loads((ROOT / 'research' / 'features.json').read_text())
    training_names = sorted(training_map)
    training = [training_map[name] for name in training_names]
    references = {'gate_mix': build_reference(training, GATE_FIELDS),
                  'shape': build_reference(training, SHAPE_FIELDS)}
    bounds = feature_bounds(training)
    oof_path = ROOT / 'research' / 'categorical_model_oof.csv'
    if not oof_path.exists():
        oof_path = ROOT / 'research' / 'full_union_model_oof.csv'
    oof = {(row['filename'], int(row['threshold'])): row
           for row in csv.DictReader(oof_path.open())}
    model = RuntimeModel()
    records = []
    failures = []
    for i, path in enumerate(files, 1):
        try:
            records.append(inspect_one(path, model, training_names, training,
                                       references, bounds, oof, thresholds,
                                       args.top_k))
        except Exception as exc:
            failures.append({'filename': qasm_name(path), 'error': str(exc)})
        if i % 25 == 0 or i == len(files):
            print(f'Inspected {i}/{len(files)} circuits', flush=True)
    write_outputs(args.out_dir, records, failures, thresholds, plot=args.plot)
    print(f'Wrote {args.out_dir / "inspection.md"} and inspection.csv/json; '
          f'{len(failures)} failures', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
