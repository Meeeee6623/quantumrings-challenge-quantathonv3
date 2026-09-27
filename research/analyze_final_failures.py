"""Audit selected v7 out-of-fold errors without changing the holdout model.

    uv run --locked python research/analyze_final_failures.py

Algorithm-like tags describe QASM patterns, not verified source labels. This
script uses labels only after grouped out-of-fold prediction, for diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research'
CAP = 14_400.0


def structural_tags(base: dict, motif: dict) -> list[str]:
    """Overlapping, deliberately tentative circuit-pattern descriptions."""
    tags = []
    if motif['motif_qaoa_score'] >= .45:
        tags.append('QAOA-like ordered motif')
    for field, label, cutoff in (
        ('fingerprint_qft', 'QFT/phase-like fingerprint', .6),
        ('fingerprint_arithmetic', 'arithmetic-like fingerprint', .5),
        ('fingerprint_graph_state', 'graph-state-like fingerprint', .7),
        ('fingerprint_random_grid', 'random-grid-like fingerprint', .7),
        ('fingerprint_variational', 'variational-like fingerprint', .5),
    ):
        if base.get(field, 0) >= cutoff:
            tags.append(label)
    if base.get('resets', 0) >= 10:
        tags.append('reset-heavy')
    if base.get('huge_fast_path', 0):
        tags.append('large-file fast path')
    return tags or ['unclassified by these patterns']


def main() -> None:
    with (OUT / 'full_union_model_oof.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    base = json.loads((OUT / 'features.json').read_text())
    motif = json.loads((OUT / 'sequence_motif_features.json').read_text())
    walks = json.loads((OUT / 'chi_walk_angle_cache.json').read_text())['tables']
    with (OUT / 'algorithm_geometry_guesses.csv').open(newline='') as handle:
        family = {row['filename']: row for row in csv.DictReader(handle)}
    assert len(rows) == 1497 and len(base) == len(motif) == len(family) == 532

    scored = []
    tagged = defaultdict(list)
    for row in rows:
        name = row['filename']
        actual = float(row['actual_s'])
        predicted = float(row['matched_basis_pred_s'])
        observed_prediction = min(predicted, CAP) if row['status'] == 'timeout' else predicted
        factor = max(observed_prediction / actual, actual / observed_prediction)
        score = max(0.0, 1.0 - abs(math.log10(observed_prediction / actual)) / 2.0)
        tags = structural_tags(base[name], motif[name])
        walk = walks[name] or {}
        rotations = walk.get('n_rotation_parsed', 0)
        near_fraction = ((walk.get('n_rotation_near_zero', 0)
                          + walk.get('n_rotation_near_pi', 0)) / rotations
                         if rotations else 0.0)
        entry = {
            'filename': name,
            'threshold': int(row['threshold']),
            'status': row['status'],
            'actual_s': actual,
            'predicted_s': predicted,
            'factor_error': factor,
            'row_score': score,
            'direction': 'under' if observed_prediction < actual else 'over',
            'n_qubits': int(base[name]['n_qubits']),
            'operations': int(base[name]['ops']),
            'qasm_characters': int(base[name]['qasm_bytes']),
            'near_basis_rotation_fraction': near_fraction,
            'structural_tags': '; '.join(tags),
            'qaoa_motif_score': motif[name]['motif_qaoa_score'],
            'shor_motif_score': motif[name]['motif_shor_score'],
            'nearest_generated_mqt_family': family[name]['guessed_mqt_family'],
            'generated_family_confidence': float(family[name]['confidence']),
            'outside_generated_reference_95pct': int(family[name]['outside_reference_95pct']),
        }
        scored.append(entry)
        for tag in tags:
            tagged[tag].append(entry)

    observed_mean = sum(item['row_score'] for item in scored) / len(scored)
    expected = json.loads((OUT / 'full_union_model_validation.json').read_text())[
        'splits']['matched']['near_basis_calibration']['score']
    assert abs(observed_mean - expected) < 1e-12
    severe = sorted((item for item in scored if item['factor_error'] > 10),
                    key=lambda item: -item['factor_error'])
    counts = Counter(item['direction'] for item in severe)
    report = {
        'basis': 'v7 matched circuit-grouped out-of-fold predictions',
        'rows': len(scored), 'circuits': len(base), 'mean_score': observed_mean,
        'greater_than_10x_rows': len(severe),
        'greater_than_10x_circuits': len({item['filename'] for item in severe}),
        'greater_than_10x_direction': dict(counts),
        'greater_than_10x_timeouts': sum(item['status'] == 'timeout' for item in severe),
        'greater_than_10x_outside_generated_reference_rows': sum(
            item['outside_generated_reference_95pct'] for item in severe),
        'tag_definitions': 'Multi-label QASM-pattern thresholds in analyze_final_failures.py; not verified algorithms.',
        'tag_segments': {
            tag: {
                'circuits': len({item['filename'] for item in items}),
                'rows': len(items),
                'mean_score': sum(item['row_score'] for item in items) / len(items),
                'greater_than_10x_rows': sum(item['factor_error'] > 10 for item in items),
            }
            for tag, items in sorted(tagged.items())
        },
    }
    path = OUT / 'final_failure_audit.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(severe[0]))
        writer.writeheader()
        writer.writerows(severe)
    (OUT / 'final_failure_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{len(severe)} severe rows / {report['greater_than_10x_circuits']} circuits; "
          f'matched score {observed_mean:.6f}')


if __name__ == '__main__':
    main()
