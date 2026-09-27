"""Reproduce the proposal's label audit; no simulator or third-party dependencies."""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit():
    rows = list(csv.DictReader((ROOT / 'runtime-data.csv').open()))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row['filename']][int(row['threshold'])] = row
    result = {
        'rows': len(rows), 'circuits': len(grouped),
        'duplicate_keys': len(rows) - sum(map(len, grouped.values())),
        'settings': {k: sorted({r[k] for r in rows})
                     for k in ('shots', 'backend', 'precision')},
        'threshold_coverage': {str(k): v for k, v in
                               Counter(tuple(sorted(g)) for g in grouped.values()).items()},
        'status_duration_mismatches': sum((r['status'] == 'timeout') !=
                                         (not r['duration_s']) for r in rows),
        'success_above_cap': [r for r in rows if r['status'] == 'success'
                              and float(r['duration_s']) > 14400],
        'by_threshold': {}, 'paired_comparisons': {},
    }
    for threshold in (16, 64, 512):
        subset = [r for r in rows if int(r['threshold']) == threshold]
        durations = [float(r['duration_s']) for r in subset if r['status'] == 'success']
        result['by_threshold'][threshold] = {
            'rows': len(subset), 'status': dict(Counter(r['status'] for r in subset)),
            'success_min': min(durations), 'success_median': statistics.median(durations),
            'success_max': max(durations),
        }
    for low, high in ((16, 64), (64, 512), (16, 512)):
        pairs = [(float(g[low]['duration_s']), float(g[high]['duration_s']))
                 for g in grouped.values() if low in g and high in g
                 and g[low]['status'] == g[high]['status'] == 'success']
        result['paired_comparisons'][f'{low}->{high}'] = {
            'both_success': len(pairs), 'higher_threshold_faster': sum(b < a for a, b in pairs),
            'higher_threshold_at_least_20pct_faster': sum(b < .8*a for a, b in pairs),
            'median_runtime_ratio': statistics.median(b/a for a, b in pairs),
        }
    files = {p.name.removesuffix('.zst') for p in (ROOT / 'training_circuits').glob('*.qasm.zst')}
    result['circuit_files'] = len(files)
    result['missing_files'] = sorted(set(grouped) - files)
    result['unlabeled_files'] = sorted(files - set(grouped))
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
