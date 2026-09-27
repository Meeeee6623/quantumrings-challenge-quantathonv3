"""Cache bounded same-parser JEPA gate tokens for the released circuits."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'quantathon-harness'))
from model import RuntimeModel  # noqa: E402
from run import read_qasm  # noqa: E402

OUTPUT = ROOT / 'research' / 'jepa_tokens.json'


def main():
    model = RuntimeModel()
    tables, timing = {}, []
    paths = sorted((ROOT / 'training_circuits').glob('*.qasm.zst'))
    for index, path in enumerate(paths, 1):
        qasm = read_qasm(path)
        started = time.perf_counter()
        # Token training needs only the main parser's gate stream; do not pay
        # for the existing independent DAG and chi-walk feature passes here.
        features = model.featurize(qasm, collect_jepa_tokens=True, run_aux_features=False)
        elapsed = time.perf_counter() - started
        name = path.name[:-4]
        tables[name] = features.pop('_jepa_tokens')
        timing.append({'filename':name, 'seconds':elapsed, 'tokens':len(tables[name]),
                       'bytes':len(qasm)})
        if index % 25 == 0 or index == len(paths):
            print(f'tokenized {index}/{len(paths)}; max={max(x["seconds"] for x in timing):.3f}s', flush=True)
    OUTPUT.write_text(json.dumps({'token_dim':8, 'tables':tables, 'timing':timing}, separators=(',', ':'))+'\n')
    print(json.dumps({'output':str(OUTPUT), 'circuits':len(tables),
                      'max_seconds':max(x['seconds'] for x in timing),
                      'zero_token_circuits':sum(not x['tokens'] for x in timing)}, indent=2))


if __name__ == '__main__':
    main()
