"""Conservative runtime floors for two rare, costly circuit regimes.

The reference bank contains only QASM-derived operation counts, thresholds, and
observed training runtimes.  It never contains filenames or family labels.
"""

from __future__ import annotations

import math


CAP_SECONDS = 14400.0


def eligible(features):
    return (features.get('resets', 0) >= 10
            and features.get('multi_q', 0) >= 10
            and features.get('fingerprint_grover', 0) >= 0.9)


def runtime_floor(features, threshold, references):
    """Scale the nearest same-threshold training analogue by gate count."""
    if not eligible(features):
        return 0.0
    ops = max(1.0, float(features.get('ops', 0)))
    candidates = [reference for reference in references
                  if int(reference[0]) == int(threshold)]
    if not candidates:
        return 0.0
    reference_threshold, reference_ops, reference_seconds = min(
        candidates, key=lambda reference: abs(math.log(
            ops / max(1.0, float(reference[1])))))
    del reference_threshold
    return min(CAP_SECONDS, min(CAP_SECONDS, float(reference_seconds))
               * ops / max(1.0, float(reference_ops)))


def apply_floor(predicted_seconds, features, threshold, references):
    return max(float(predicted_seconds), runtime_floor(features, threshold, references))


def large_work_floor(features):
    """Minimum observed work scale for circuits with many active operations.

    These are deliberately below the smallest observed runtimes in the two
    operation-count regimes; the floor only corrects severe extrapolation.
    """
    effective_ops = float(features.get('effective_ops', 0))
    if effective_ops >= 1_000_000:
        return 100.0
    if effective_ops >= 500_000:
        return 10.0
    return 0.0
