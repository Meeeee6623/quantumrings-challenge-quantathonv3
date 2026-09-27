"""Label-free redundancy audit and fold-local feature selection."""
from __future__ import annotations

import numpy as np


def redundant_columns(X, columns):
    """Return retained positions and an audit of constant/exact duplicate columns."""
    seen = {}
    retained = []
    constant = []
    duplicate = []
    for j, name in enumerate(columns):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if not len(finite) or (len(finite) == len(col) and finite.min() == finite.max()):
            constant.append(name)
            continue
        key = col.tobytes()
        if key in seen:
            duplicate.append({'removed': name, 'kept': seen[key]})
            continue
        seen[key] = name
        retained.append(j)
    return np.asarray(retained, dtype=int), constant, duplicate


def importance_choice(estimator, X, y, limit):
    """Fit a ranker on training rows and return original-order top positions."""
    if limit is None or limit >= X.shape[1]:
        return np.arange(X.shape[1])
    estimator.fit(X, y)
    rank = np.argsort(-estimator.feature_importances_, kind='stable')
    return np.sort(rank[:limit])
