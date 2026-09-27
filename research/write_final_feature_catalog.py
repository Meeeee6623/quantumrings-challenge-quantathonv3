"""Write the exact fitted v7 feature list and component membership.

    uv run --locked python research/write_final_feature_catalog.py

The CSV is generated from the serialized submission artifact, not from an
unfitted candidate list or a stale research note.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'quantathon-harness' / 'artifacts' / 'runtime_model.joblib'
REPORT = ROOT / 'research' / 'full_union_model_validation.json'
OUT = ROOT / 'research' / 'final_feature_catalog.csv'


def view(name: str) -> str:
    if name.startswith('extended__'):
        return 'secondary QASM scanner'
    if name.startswith('chi_walk_'):
        return 'angle-aware chi walk'
    if name in ('threshold', 'log_threshold'):
        return 'simulator setting'
    if name.startswith('log_'):
        return 'log1p primary-parser transform'
    return 'primary QASM parser'


def main() -> None:
    artifact = joblib.load(ARTIFACT)
    schema = json.loads(REPORT.read_text())['selected_schema']
    names = artifact['columns']
    assert names == schema['selected_columns']
    roles = {'global_runtime': set(artifact['global_columns'])}
    for threshold in (16, 64, 512):
        roles[f'runtime_{threshold}'] = set(
            artifact['specialist_columns_by_threshold'][threshold])
        roles[f'timeout_{threshold}'] = set(
            artifact['classifier_columns_by_threshold'][threshold])
    assert set(names) == set().union(*roles.values())
    assert len(names) == 325 and len(roles['global_runtime']) == 120
    assert all(len(roles[f'runtime_{t}']) == 200 and
               len(roles[f'timeout_{t}']) == 80 for t in (16, 64, 512))
    with OUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ordinal', 'feature', 'view',
                                                     *roles])
        writer.writeheader()
        for ordinal, name in enumerate(names, start=1):
            writer.writerow({'ordinal': ordinal, 'feature': name,
                             'view': view(name),
                             **{role: int(name in columns)
                                for role, columns in roles.items()}})
    print(f'Wrote {len(names)} exact selected feature names and seven role flags to {OUT}')


if __name__ == '__main__':
    main()
