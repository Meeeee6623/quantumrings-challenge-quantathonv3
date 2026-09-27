"""Bounded gate-token collection shared by JEPA training and submission inference.

The collector is deliberately dependency-free and is called from the existing
QASM scan.  It keeps a deterministic reservoir in each of the parser's eight
source-order windows, so embedding a circuit never requires a second scan or
an unbounded allocation for generated QASM files.
"""
from __future__ import annotations

import math

MAX_TOKENS_PER_WINDOW = 64
TOKEN_DIM = 8

_FAMILIES = {
    'id': 1, 'i': 1, 'x': 2, 'y': 2, 'z': 3, 'h': 4, 's': 5, 'sdg': 5,
    't': 6, 'tdg': 6, 'sx': 7, 'sxdg': 7,
    'rx': 8, 'ry': 9, 'rz': 10, 'p': 10, 'u1': 10, 'u': 11, 'u2': 11,
    'u3': 11, 'cx': 12, 'cy': 12, 'cz': 13, 'ch': 12, 'cp': 13,
    'crx': 14, 'cry': 14, 'crz': 14, 'swap': 15, 'iswap': 15, 'ecr': 15,
    'rxx': 16, 'ryy': 16, 'rzz': 16, 'ccx': 17, 'ccz': 17, 'cswap': 17,
    'measure': 18, 'reset': 19, 'barrier': 20, 'delay': 21,
}
FAMILY_COUNT = 24


def _angle_bucket(angle: float | None) -> int:
    if angle is None:
        return 0
    wrapped = math.remainder(angle, 2 * math.pi)
    if abs(wrapped) < 1e-9:
        return 1
    # Preserve coarse rotation class without depending on arbitrary syntax.
    return 2 + int((wrapped + math.pi) / (2 * math.pi) * 14) % 14


class JEPATokenCollector:
    """Collect at most 512 source-order gate descriptors without randomness."""

    def __init__(self, per_window: int = MAX_TOKENS_PER_WINDOW):
        self.per_window = per_window
        self.seen = [0] * 8
        self.items: list[list[tuple[int, tuple[int, int, int, int, int, int]]]] = [[] for _ in range(8)]

    def add(self, gate: str, qubits: list[int], angle: float | None, window: int) -> None:
        window = min(7, max(0, int(window)))
        self.seen[window] += 1
        order = self.seen[window]
        arity = min(3, len(qubits))
        if qubits:
            q0, q1 = min(qubits), max(qubits)
        else:
            q0 = q1 = 0
        item = (order, (_FAMILIES.get(gate, 0), arity, _angle_bucket(angle),
                        q0, q1, int(gate in ('cx', 'cy', 'cz', 'cp', 'swap', 'iswap',
                                               'ecr', 'rxx', 'ryy', 'rzz', 'ccx', 'ccz'))))
        bucket = self.items[window]
        if len(bucket) < self.per_window:
            bucket.append(item)
            return
        # A deterministic reservoir avoids filename or RNG-dependent features.
        slot = ((order * 1103515245 + 12345 + 97 * window) & 0x7fffffff) % order
        if slot < self.per_window:
            bucket[slot] = item

    def matrix(self, n_qubits: int) -> list[list[float]]:
        scale = max(1, n_qubits - 1)
        rows: list[list[float]] = []
        for window, bucket in enumerate(self.items):
            for _, (family, arity, angle, q0, q1, is_two) in sorted(bucket):
                rows.append([
                    family / FAMILY_COUNT,
                    arity / 3.0,
                    angle / 15.0,
                    window / 7.0,
                    min(1.0, q0 / scale),
                    min(1.0, q1 / scale),
                    min(1.0, abs(q1 - q0) / scale),
                    float(is_two),
                ])
        return rows
