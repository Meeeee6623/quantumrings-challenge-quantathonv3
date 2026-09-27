"""Small latent-prediction encoder used as an optional runtime feature source."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from jepa_tokens import TOKEN_DIM

EMBEDDING_DIM = 32


def torch_modules():
    import torch
    from torch import nn

    class CircuitEncoder(nn.Module):
        def __init__(self, hidden: int = 64, embedding_dim: int = EMBEDDING_DIM):
            super().__init__()
            self.input = nn.Linear(TOKEN_DIM, hidden)
            self.position = nn.Embedding(512, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                dropout=0.0, batch_first=True, activation='gelu', norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
            self.norm = nn.LayerNorm(hidden)
            self.output = nn.Linear(hidden, embedding_dim)
            self.mask_token = nn.Parameter(torch.zeros(hidden))

        def forward(self, tokens, valid, masked=None, return_tokens=False):
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = self.input(tokens) + self.position(positions)[None, :, :]
            if masked is not None:
                x = torch.where(masked[..., None], self.mask_token, x)
            x = self.transformer(x, src_key_padding_mask=~valid)
            x = self.norm(x)
            token_embeddings = self.output(x)
            denom = valid.sum(1, keepdim=True).clamp(min=1)
            pooled = (token_embeddings * valid[..., None]).sum(1) / denom
            return (pooled, token_embeddings) if return_tokens else pooled

    class Predictor(nn.Module):
        def __init__(self, dim: int = EMBEDDING_DIM):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        def forward(self, x):
            return self.net(x)

    return CircuitEncoder, Predictor


def padded_batch(rows: list[list[list[float]]], device):
    import torch
    max_len = max(1, max((len(row) for row in rows), default=0))
    tokens = torch.zeros((len(rows), max_len, TOKEN_DIM), dtype=torch.float32, device=device)
    valid = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        if row:
            values = torch.tensor(row, dtype=torch.float32, device=device)
            tokens[i, :len(row)] = values
            valid[i, :len(row)] = True
        else:
            # Transformer attention cannot accept an all-padding sequence.
            # Callers zero this sentinel embedding before handing it to trees.
            valid[i, 0] = True
    return tokens, valid


def mask_spans(valid, generator):
    """Mask one contiguous temporal region per circuit, at least one token."""
    import torch
    masked = torch.zeros_like(valid)
    for i, length in enumerate(valid.sum(1).tolist()):
        if not length:
            continue
        width = max(1, round(length * 0.25))
        start = int(torch.randint(0, max(1, length - width + 1), (1,), generator=generator).item())
        masked[i, start:start + width] = True
    return masked & valid


def load_encoder(path: Path, device='cpu'):
    import torch
    CircuitEncoder, _ = torch_modules()
    payload = torch.load(path, map_location=device, weights_only=True)
    encoder = CircuitEncoder(hidden=int(payload['hidden']), embedding_dim=int(payload['embedding_dim']))
    encoder.load_state_dict(payload['state_dict'])
    encoder.eval().to(device)
    return encoder, payload


def embed_rows(encoder, rows: list[list[list[float]]], device='cpu') -> np.ndarray:
    import torch
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            tokens, valid = padded_batch(rows[start:start + 32], device)
            output.append(encoder(tokens, valid).detach().cpu().numpy())
    result = np.concatenate(output, axis=0).astype(np.float64, copy=False)
    for i, row in enumerate(rows):
        if not row:
            result[i] = 0.0
    return result
