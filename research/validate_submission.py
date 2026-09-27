"""Validate complete holdout CSV coverage and the harness timing caps.

    uv run --locked python research/validate_submission.py \
      --circuits path/to/holdout --submission submission.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from inspect_holdout import circuit_files, qasm_name

PARSE_CAP_S = 15.0
PREDICT_CAP_S = 15.0


def validate(circuits: Path, submission: Path, thresholds: list[int]):
    files = circuit_files(circuits)
    expected = {(qasm_name(path), threshold) for path in files
                for threshold in thresholds}
    with submission.open(newline='') as file:
        reader = csv.DictReader(file)
        required = {'team', 'filename', 'threshold', 'pred_duration_s',
                    'parse_s', 'predict_s'}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            return {'valid': False, 'issues': [
                f'Missing columns: {sorted(missing_columns)}']}
        rows = list(reader)
    seen = set()
    issues = []
    predictions = []
    parses = []
    inference = []
    teams = set()
    for i, row in enumerate(rows, 2):
        try:
            key = row['filename'], int(row['threshold'])
            prediction = float(row['pred_duration_s'])
            parse_s = float(row['parse_s'])
            predict_s = float(row['predict_s'])
            assert math.isfinite(prediction) and prediction > 0
            assert math.isfinite(parse_s) and parse_s >= 0
            assert math.isfinite(predict_s) and predict_s >= 0
        except (ValueError, TypeError, AssertionError) as exc:
            issues.append(f'Invalid numeric value on CSV line {i}: {exc}')
            continue
        teams.add(row['team'])
        if key in seen:
            issues.append(f'Duplicate prediction: {key}')
        seen.add(key)
        predictions.append(prediction)
        parses.append(parse_s)
        inference.append(predict_s)
        if parse_s > PARSE_CAP_S:
            issues.append(f'Parse cap exceeded: {key} ({parse_s:.4f}s)')
        if predict_s > PREDICT_CAP_S:
            issues.append(f'Predict cap exceeded: {key} ({predict_s:.4f}s)')
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing:
        issues.append(f'Missing predictions ({len(missing)}): {missing[:10]}')
    if unexpected:
        issues.append(f'Unexpected predictions ({len(unexpected)}): {unexpected[:10]}')
    if not teams or any(not name.strip() for name in teams) or len(teams) != 1:
        issues.append(f'Expected one nonempty team name; found {sorted(teams)}')
    return {'valid': not issues, 'circuits': len(files),
            'expected_rows': len(expected), 'csv_rows': len(rows),
            'valid_numeric_rows': len(predictions),
            'max_parse_s': max(parses, default=None),
            'max_predict_s': max(inference, default=None),
            'min_prediction_s': min(predictions, default=None),
            'max_prediction_s': max(predictions, default=None),
            'issues': issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuits', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--thresholds', default='16,64,512')
    args = parser.parse_args()
    thresholds = [int(item) for item in args.thresholds.split(',')]
    if not thresholds or len(thresholds) != len(set(thresholds)):
        parser.error('Thresholds must be a nonempty comma-separated unique list')
    result = validate(args.circuits, args.submission, thresholds)
    for key, value in result.items():
        if key != 'issues':
            print(f'{key}: {value}')
    for issue in result['issues']:
        print(f'ERROR: {issue}')
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
