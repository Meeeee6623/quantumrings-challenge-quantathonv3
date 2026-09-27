"""Threshold transforms and empirically motivated interaction terms."""

from __future__ import annotations

import math
from collections.abc import Mapping


def threshold_features(threshold: float, features: Mapping[str, float]) -> dict[str, float]:
    value = float(threshold)
    valid = value > 0 and math.isfinite(value)
    result = {
        "threshold": value,
        "threshold_log": math.log(value) if valid else math.nan,
        "threshold_log2": math.log2(value) if valid else math.nan,
        "threshold_neg_log10": -math.log10(value) if valid else math.nan,
        "threshold_is_16": float(math.isclose(value, 16.0)),
        "threshold_is_64": float(math.isclose(value, 64.0)),
        "threshold_is_512": float(math.isclose(value, 512.0)),
    }
    interactions = {
        "num_qubits": "num_qubits",
        "gate_count": "total_gate_count",
        "depth": "circuit_depth",
        "entanglement_ratio": "supermarq_entanglement_ratio",
        "approx_treewidth": "approx_treewidth",
        "expanded_gate_count": "custom_expanded_gate_count_proxy",
        "cut_crossings_max": "cut_crossings_max",
        "interaction_spectral_radius": "interaction_spectral_radius",
        "angle_generic_fraction": "angle_generic_fraction",
    }
    for label, feature_name in interactions.items():
        feature_value = float(features.get(feature_name, math.nan))
        result[f"threshold_x_{label}"] = value * feature_value
        result[f"threshold_log_x_{label}"] = math.log2(value) * feature_value if valid else math.nan
    return result

