"""Conservative high-threshold calibration for near-basis rotations."""

from __future__ import annotations


CAP_SECONDS = 14400.0


def calibrate_near_basis_runtime(seconds, features, threshold):
    """Reduce overpredictions in large near-basis circuits at threshold 512.

    The chi walk marks rotations close to integer multiples of pi.  The guard
    preserves low-latency predictions and explicit timeout predictions.
    """
    seconds = float(seconds)
    if (int(threshold) == 512
            and features.get('chi_walk_rot_near_frac',0) >= 0.9
            and features.get('ops',0) >= 1000
            and 1.0 <= seconds < CAP_SECONDS):
        return seconds * 0.5
    return seconds
