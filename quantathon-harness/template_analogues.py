"""Label-free structural signature with a cautious training analogue blend."""

from __future__ import annotations

import math
import statistics


SIGNATURE_FIELDS = ('n_qubits','ops','two_q','multi_q')


def signature(features):
    return tuple(int(float(features.get(field,0))) for field in SIGNATURE_FIELDS)


def blend_with_analogues(predicted_seconds, features, threshold, bank):
    """Blend only when two or more same-threshold templates are available."""
    log_seconds = bank.get((int(threshold),*signature(features)),())
    if len(log_seconds) < 2:
        return float(predicted_seconds)
    analogue = 10**statistics.median(log_seconds)
    return math.sqrt(float(predicted_seconds)*analogue)
