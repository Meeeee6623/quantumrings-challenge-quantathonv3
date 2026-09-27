"""Build the second-pass structural/DAG feature cache for union training.

The extractor is also embedded in ``quantathon-harness`` for holdout inference.
This cache only avoids rescanning the 532 training files on every experiment.

    uv run --locked python research/extract_extended_features.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'quantathon-harness'))
from extended_features import extract_fast_qasm_features  # noqa: E402
from run import read_qasm  # noqa: E402

COLUMNS = ROOT / 'research' / 'extended_feature_columns.json'
OUTPUT = ROOT / 'research' / 'extended_features.json'


def safe_value(value):
    value = float(value)
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--circuits',type=Path,default=ROOT / 'training_circuits')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    columns = json.loads(COLUMNS.read_text())
    circuit_columns = [column for column in columns if column != 'threshold']
    paths = sorted(args.circuits.glob('*.qasm.zst'))
    if not paths:
        raise SystemExit(f'no .qasm.zst circuits under {args.circuits}')
    tables = {}
    timing = []
    for index,path in enumerate(paths,1):
        qasm = read_qasm(path)
        started = time.perf_counter()
        extracted = extract_fast_qasm_features(qasm)
        elapsed = time.perf_counter()-started
        name = path.name[:-4]
        tables[name] = {column:safe_value(extracted.get(column,math.nan))
                        for column in circuit_columns}
        timing.append({'filename':name,'seconds':elapsed,'bytes':len(qasm)})
        if index % 25 == 0 or index == len(paths):
            print(f'extracted {index}/{len(paths)}; slowest={max(x["seconds"] for x in timing):.3f}s',
                  flush=True)
    payload = {'columns':columns,'tables':tables,'timing':timing}
    args.output.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    print(f'wrote {args.output}; max={max(x["seconds"] for x in timing):.3f}s')


if __name__ == '__main__':
    main()
