"""Fold-isolated masked latent prediction training for bounded QASM tokens."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'quantathon-harness'))
from jepa_encoder import EMBEDDING_DIM, embed_rows, mask_spans, padded_batch, torch_modules  # noqa: E402

TOKENS = ROOT / 'research' / 'jepa_tokens.json'


def load_tokens(path=TOKENS):
    return json.loads(path.read_text())['tables']


def device_for_torch(torch):
    return 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'


def fit_encoder(tokens_by_name, training_names, epochs=6, seed=17, batch_size=16):
    """Fit a small EMA target/context encoder using QASM only, never labels."""
    import torch
    import torch.nn.functional as F

    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device_for_torch(torch)
    CircuitEncoder, Predictor = torch_modules()
    encoder = CircuitEncoder(hidden=64, embedding_dim=EMBEDDING_DIM).to(device)
    target = deepcopy(encoder).eval().to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    predictor = Predictor().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()),
                                  lr=1e-3, weight_decay=1e-4)
    names = sorted(name for name in set(training_names) if tokens_by_name[name])
    rows = [tokens_by_name[name] for name in names]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    losses = []
    encoder.train()
    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = [rows[i] for i in order[start:start + batch_size]]
            tokens, valid = padded_batch(batch, device)
            mask = mask_spans(valid, generator)
            context = encoder(tokens, valid, mask)
            with torch.no_grad():
                _, target_tokens = target(tokens, valid, return_tokens=True)
                denom = mask.sum(1, keepdim=True).clamp(min=1)
                expected = (target_tokens * mask[..., None]).sum(1) / denom
            predicted = predictor(context)
            alignment = F.mse_loss(F.normalize(predicted, dim=1), F.normalize(expected, dim=1))
            # A small variance term prevents the trivial all-circuits embedding.
            std = context.std(dim=0, unbiased=False)
            variance = F.relu(0.05 - std).mean()
            loss = alignment + 0.1 * variance
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), 1.0)
            optimizer.step()
            with torch.no_grad():
                for online, ema in zip(encoder.parameters(), target.parameters()):
                    ema.mul_(0.996).add_(online, alpha=0.004)
            losses.append(float(loss.detach().cpu()))
    encoder.eval()
    return encoder, {'device':device, 'epochs':epochs, 'examples':len(names),
                     'mean_loss':float(np.mean(losses[-20:])) if losses else None}


def embeddings_for(encoder, tokens_by_name, names, device='cpu'):
    names = list(names)
    vector = embed_rows(encoder, [tokens_by_name[name] for name in names], device)
    return dict(zip(names, vector))


def save_encoder(encoder, path):
    import torch
    payload = {'hidden':64, 'embedding_dim':EMBEDDING_DIM, 'state_dict':encoder.cpu().state_dict()}
    torch.save(payload, path)
