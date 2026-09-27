"""Independent runtime-only test of Xing et al. (arXiv:2606.11620v1).

The source paper's fidelity-derived threshold target and four execution
contexts are unavailable here. This script implements its runtime architecture
and ablations on the Quantum Rings challenge's circuit-grouped folds.

    uv run --locked --extra paper python research/paper_runtime_replication.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_full_union_model import ROOT, load_fold_table, load_training_table
from train_merged_model import CAP, metrics, paired_bootstrap

OUT = ROOT / 'research'
FEATURES = OUT / 'features.json'
EXTENDED = OUT / 'extended_features.json'
FAMILY_PROBS = OUT / 'algorithm_geometry_predictions.json'
OOF = OUT / 'paper_runtime_oof.csv'
REPORT = OUT / 'paper_runtime_replication.json'
GATE_NAMES = ('h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap')
GRAPH_COUNT = 5
PAPER_DIM = 32


def paper_features(base: dict, extended: dict, threshold: int):
    """32 paper-category proxies plus the simulator threshold (33 total)."""
    n = max(1.0, float(base.get('n_qubits', 0)))
    ops = max(1.0, float(base.get('ops', 0)))
    basic = [base.get('depth', 0), base.get('n_qubits', 0), base.get('ops', 0)]
    gates = [extended.get(f'gate_count__{gate}', 0) for gate in GATE_NAMES]
    # The challenge uses one opaque backend and single precision; its CPU/GPU
    # indicator is unknown, not silently inferred from the backend name.
    context = [0.0, 0.0, 1.0, 0.0]
    complexity = [float(base.get('two_q', 0)) / ops,
                  base.get('graph_density', 0),
                  float(base.get('depth', 0)) / n]
    local = [base.get('max_degree', 0),
             base.get('graph_degree_entropy', 0)]
    fingerprints = [base.get('fingerprint_arithmetic', 0),
                    base.get('fingerprint_qft', 0),
                    base.get('fingerprint_variational', 0)]
    graph = [extended.get('interaction_clustering_coefficient', 0),
             base.get('components', 0),
             base.get('graph_cutwidth_rcm', 0),
             base.get('graph_mean_cut_rcm', 0),
             base.get('graph_span_rcm', 0)]
    vector = basic + gates + context + complexity + local + fingerprints + graph
    assert len(vector) == PAPER_DIM
    # Paper Section IV-E implies threshold enters runtime prediction, although
    # its 32-feature table omits it. We add it explicitly for this adaptation.
    vector += [math.log2(max(1, threshold))]
    bounded = {15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 32}
    for i, value in enumerate(vector):
        value = float(value or 0)
        if i not in bounded:
            value = math.log1p(max(0, value))
        vector[i] = value
    return np.nan_to_num(np.asarray(vector, dtype=np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)


def load_paper_table():
    rows, _, _, y, timeout, thresholds, names = load_training_table()
    base = json.loads(FEATURES.read_text())
    extended = json.loads(EXTENDED.read_text())['tables']
    probabilities = json.loads(FAMILY_PROBS.read_text())
    families = sorted({family for item in probabilities.values()
                       for family in item})
    X = np.vstack([paper_features(base[row['filename']],
                                   extended[row['filename']], int(row['threshold']))
                   for row in rows])
    family = np.asarray([families.index(max(
        probabilities[name], key=probabilities[name].get)) for name in names],
        dtype=np.int64)
    return rows, X, np.asarray(y), np.asarray(timeout), np.asarray(thresholds), names, family, families


class PaperRuntimeNet(nn.Module):
    """Shared SiLU backbone; optional concat or FiLM + additive residual."""
    def __init__(self, width: int, families: int, mode: str):
        super().__init__()
        if mode not in ('plain', 'concat', 'family'):
            raise ValueError(mode)
        self.mode = mode
        self.families = families
        incoming = width + (families if mode == 'concat' else 0)
        self.global_processor = nn.Sequential(nn.Linear(incoming, 64), nn.SiLU(),
                                              nn.Dropout(.2))
        self.backbone = nn.Sequential(nn.Linear(64, 64), nn.SiLU(),
                                      nn.Dropout(.2), nn.Linear(64, 64),
                                      nn.SiLU(), nn.Dropout(.2))
        self.runtime_head = nn.Linear(64, 1)
        self.shortcut = nn.Linear(incoming, 1)
        if mode == 'family':
            self.embedding = nn.Embedding(21, 64)
            self.family_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(),
                                            nn.Dropout(.2), nn.Linear(64, 64),
                                            nn.SiLU(), nn.Dropout(.2))
            # Concatenating family and global streams before the first
            # backbone layer is equivalent to adding this separate projection.
            # Zero initialization preserves the paper's family-agnostic start.
            self.family_to_backbone = nn.Linear(64, 64, bias=False)
            self.gamma = nn.Linear(64, 64)
            self.beta = nn.Linear(64, 64)
            self.family_residual = nn.Linear(64, 1)
            for layer in (self.family_to_backbone, self.gamma, self.beta,
                          self.family_residual):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, family):
        if self.mode == 'concat':
            x = torch.cat((x, nn.functional.one_hot(
                family, num_classes=self.families).float()), dim=1)
        global_features = self.global_processor(x)
        family_offset = 0.0
        if self.mode == 'family':
            f = self.family_mlp(self.embedding(family))
            global_features = global_features + self.family_to_backbone(f)
        h = self.backbone(global_features)
        if self.mode == 'family':
            h = (1 + self.gamma(f)) * h + self.beta(f)
            family_offset = self.family_residual(f)
        return (self.runtime_head(h) + self.shortcut(x) + family_offset).squeeze(-1)


def inner_split(train_indices, names, seed):
    unique = np.asarray(sorted(set(names[train_indices])))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    valid_names = set(unique[:max(1, int(.15 * len(unique)))])
    valid = np.asarray([i for i in train_indices if names[i] in valid_names])
    fit = np.asarray([i for i in train_indices if names[i] not in valid_names])
    assert not set(names[fit]) & set(names[valid])
    return fit, valid


def train_fold(X, y, family, names, train_indices, test_indices, mode, seed,
               max_epochs=200, patience=50):
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    fit, valid = inner_split(train_indices, np.asarray(names), seed)
    scaler = StandardScaler().fit(X[fit])
    data = torch.as_tensor(scaler.transform(X), dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.float32)
    family_t = torch.as_tensor(family, dtype=torch.long)
    model = PaperRuntimeNet(X.shape[1], int(family.max()) + 1, mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.001)
    def lr(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        progress = (epoch - 10) / max(1, max_epochs - 10)
        return .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr)
    generator = torch.Generator().manual_seed(seed)
    batches = DataLoader(TensorDataset(data[fit], family_t[fit], targets[fit]),
                         batch_size=32, shuffle=True, generator=generator)
    best = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, fb, yb in batches:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb, fb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.mse_loss(
                model(data[valid], family_t[valid]), targets[valid]))
        if validation_loss < best - 1e-6:
            best = validation_loss
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(data[test_indices], family_t[test_indices]).numpy()
    return predicted, {'epochs': epoch + 1, 'best_epoch': best_epoch,
                       'validation_log_mse': best}


def evaluate(y, seconds, timeout, thresholds):
    actual = np.power(10, y)
    # A timeout label is right-censored at the challenge's 4-hour cap.
    # Score and auxiliary R² diagnostics use the same observed-duration rule.
    observed_prediction = np.where(timeout, np.minimum(seconds, CAP), seconds)
    return {**metrics(y, timeout, thresholds, seconds),
            'r2_log10': float(r2_score(y, np.log10(observed_prediction))),
            'r2_seconds': float(r2_score(actual, observed_prediction)),
            'median_relative_error': float(np.median(np.abs(observed_prediction / actual - 1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='Two folds and 5 epochs for pipeline verification')
    args = parser.parse_args()
    rows, X, y, timeout, thresholds, names, family, families = load_paper_table()
    folds = load_fold_table()
    baseline = list(csv.DictReader((OUT / 'full_union_model_oof.csv').open()))
    assert [(r['filename'], int(r['threshold'])) for r in rows] == [
        (r['filename'], int(r['threshold'])) for r in baseline]
    report = {'source': 'Xing et al., arXiv:2606.11620v1',
              'scope': 'runtime-only adaptation on Quantum Rings challenge; not exact paper dataset reproduction',
              'families': families, 'family_source': 'frozen eight-family MQT geometry classifier (96.9% size-grouped accuracy on its own generated references)',
              'feature_count': int(X.shape[1]), 'rows': len(rows),
              'circuits': len(set(names)), 'splits': {}}
    saved = {}
    for split in ('matched', 'structural'):
        key = 'matched_fold' if split == 'matched' else 'structural_stress_fold'
        assigned = np.asarray([int(folds[str(name)][key]) for name in names])
        predictions = {}
        folds_to_run = sorted(set(assigned))[:2] if args.smoke else sorted(set(assigned))
        for label, mode, mask in (
                ('mlp_basic', 'plain', list(range(27)) + [32]),
                ('mlp_graph', 'plain', list(range(33))),
                ('family_concat', 'concat', list(range(33))),
                ('family_film_residual', 'family', list(range(33)))):
            output = np.full(len(rows), np.nan)
            logs = []
            for fold in folds_to_run:
                train = np.flatnonzero(assigned != fold)
                test = np.flatnonzero(assigned == fold)
                predicted, detail = train_fold(
                    X[:, mask], y, family, names, train, test, mode,
                    seed=1700 + fold, max_epochs=5 if args.smoke else 200,
                    patience=3 if args.smoke else 50)
                output[test] = predicted
                logs.append({'fold': int(fold), **detail})
                print(split, label, fold, 'best_epoch', detail['best_epoch'],
                      flush=True)
            predictions[label] = np.power(10, np.clip(output, -8, 8))
            if not args.smoke:
                report['splits'].setdefault(split, {})[label] = {
                    **evaluate(y, predictions[label], timeout, thresholds),
                    'training': logs}
        if args.smoke:
            continue
        current = np.asarray([float(row[f'{split}_basis_pred_s']) for row in baseline])
        report['splits'][split]['current_v7'] = evaluate(y, current, timeout, thresholds)
        report['splits'][split]['family_vs_graph'] = paired_bootstrap(
            y, timeout, names, folds, split,
            predictions['mlp_graph'], predictions['family_film_residual'])
        report['splits'][split]['graph_vs_basic'] = paired_bootstrap(
            y, timeout, names, folds, split,
            predictions['mlp_basic'], predictions['mlp_graph'])
        for label, seconds in predictions.items():
            saved[(split, label)] = seconds
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(split, json.dumps({label: report['splits'][split][label]['score']
                                 for label in ('mlp_basic', 'mlp_graph',
                                               'family_concat',
                                               'family_film_residual', 'current_v7')}),
              flush=True)
    if args.smoke:
        print('Smoke fit completed; no final result files written.')
        return
    with OOF.open('w', newline='') as file:
        labels = ('mlp_basic', 'mlp_graph', 'family_concat', 'family_film_residual')
        fields = ['filename', 'threshold', 'status', 'actual_s', 'predicted_family']
        fields += [f'{split}_{label}_pred_s' for split in ('matched', 'structural')
                   for label in labels]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            item = {'filename': row['filename'], 'threshold': row['threshold'],
                    'status': row['status'], 'actual_s': 10 ** y[i],
                    'predicted_family': families[family[i]]}
            item.update({f'{split}_{label}_pred_s': saved[(split, label)][i]
                         for split in ('matched', 'structural')
                         for label in labels})
            writer.writerow(item)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote', REPORT, OOF, flush=True)


if __name__ == '__main__':
    main()
