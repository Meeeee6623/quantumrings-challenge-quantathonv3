"""Ablate source-level pi syntax as two extra threshold-expert features.

This is an experiment only: it never overwrites the selected fitted artifact
or fixed-fold OOF file. Run with uv from the repository root.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'research'
sys.path[:0] = [str(ROOT / 'quantathon-harness'),str(HERE)]
from chi_walk import model_features  # noqa: E402
from run import read_qasm  # noqa: E402
from train_full_union_model import (  # noqa: E402
    cross_predict, load_extended_matrix, load_fold_table, load_training_table,
)
from train_merged_model import metrics  # noqa: E402


CACHE = HERE / 'pi_syntax_features.json'
REPORT = HERE / 'pi_syntax_probe.json'
OOF = HERE / 'pi_syntax_oof.csv'


def build_cache(names, extended):
    if CACHE.exists():
        values = json.loads(CACHE.read_text())
        if all(name in values for name in names):
            return values
    values = {}
    for i,name in enumerate(sorted(names),1):
        path = ROOT / 'training_circuits' / (name+'.zst')
        if not path.exists():
            path = ROOT / 'training_circuits' / name
        qasm = read_qasm(path)
        count = int(extended[name]['source_pi_token_count'])
        expressions = max(1,float(extended[name].get('angle_expression_count')
                                  or extended[name]['source_semicolon_count']))
        values[name] = {
            'pi_literal_fraction':count / expressions,
            'pi_tail_fraction':qasm[int(.8*len(qasm)):].lower().count('pi') / max(1,count),
        }
        if i%100==0:
            print('scanned',i,'of',len(names),flush=True)
    CACHE.write_text(json.dumps(values,indent=2)+'\n')
    return values


def main():
    rows, base_X, base_columns, y, timeout, thresholds, names = load_training_table()
    extended_X, extended_columns = load_extended_matrix(rows,thresholds)
    old_X = np.column_stack((base_X,extended_X))
    extended = json.loads((HERE / 'extended_features.json').read_text())['tables']
    new = build_cache(set(names),extended)
    features = json.loads((HERE / 'features.json').read_text())
    walks = json.loads((HERE / 'chi_walk_angle_cache.json').read_text())['tables']
    for name,walk in walks.items():
        features[name]['chi_walk_rot_near_frac']=model_features(walk)['chi_walk_rot_near_frac']
    folds = load_fold_table()
    with (HERE / 'full_union_model_oof.csv').open(newline='') as file:
        baseline = list(csv.DictReader(file))
    assert [(r['filename'],int(r['threshold'])) for r in rows] == [
        (r['filename'],int(r['threshold'])) for r in baseline]
    report = {'columns':['pi_literal_fraction','pi_tail_fraction'],
              'splits':{}}
    predictions = {}
    for columns in (('pi_literal_fraction',),
                    ('pi_literal_fraction','pi_tail_fraction')):
        extra = np.asarray([[new[name][column] for column in columns] for name in names])
        union_X = np.column_stack((old_X,extra))
        key = '+'.join(columns)
        report['splits'][key]={}
        predictions[key]={}
        for split in ('matched','structural'):
            pred = cross_predict(base_X,union_X,y,timeout,thresholds,names,rows,
                                 features,folds,split)
            candidate = pred['basis_adjusted_seconds']
            predictions[key][split] = candidate
            previous = np.asarray([float(row[f'{split}_basis_pred_s']) for row in baseline])
            result = {
                'baseline':metrics(y,timeout,thresholds,previous),
                'candidate':metrics(y,timeout,thresholds,candidate),
            }
            result['delta'] = result['candidate']['score']-result['baseline']['score']
            report['splits'][key][split] = result
            print(key,split,round(result['delta'],6),flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    with OOF.open('w',newline='') as file:
        fields = ['filename','threshold','status','actual_s']
        fields += [f'{split}_baseline_s' for split in ('matched','structural')]
        for key in predictions:
            fields += [f'{split}_{key}_s' for split in ('matched','structural')]
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for i,row in enumerate(rows):
            entry={field:row[field] for field in ('filename','threshold','status')}
            entry['actual_s']=float(baseline[i]['actual_s'])
            for split in ('matched','structural'):
                entry[f'{split}_baseline_s']=float(baseline[i][f'{split}_basis_pred_s'])
            for key in predictions:
                for split in ('matched','structural'):
                    entry[f'{split}_{key}_s']=float(predictions[key][split][i])
            writer.writerow(entry)


if __name__ == '__main__':
    main()
