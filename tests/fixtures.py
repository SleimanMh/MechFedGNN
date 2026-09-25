"""Shared test fixtures (CLAUDE.md §13)."""
import numpy as np

X_, Z_, Y_ = 0, 1, 2   # feature order [X, Z, Y]


def mask_a():
    """50 rows fully observed, 50 rows with X and Z missing together; Y always observed."""
    M = np.ones((100, 3), dtype=np.int8)
    M[50:, [X_, Z_]] = 0
    return M


def mask_b():
    """J[X,Z] = 0.9: 90 rows fully observed, 10 rows with only X missing."""
    M = np.ones((100, 3), dtype=np.int8)
    M[90:, X_] = 0
    return M


def mask_c():
    """J[X,Z] = 0.3: 30 rows fully observed, 70 rows with X and Z missing."""
    M = np.ones((100, 3), dtype=np.int8)
    M[30:, [X_, Z_]] = 0
    return M


def synthetic(n=4000, seed=0):
    """8 features: 0-3 maskable in two correlated pairs, 4-7 always observed."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]
    X[:, 3] = 0.8 * X[:, 2] + 0.6 * X[:, 3]
    y = X[:, 4:].sum(1) + 0.3 * X[:, :4].sum(1) + 0.5 * rng.standard_normal(n)
    roles = {
        "features": [f"f{i}" for i in range(8)],
        "constant": [],
        "maskable": [0, 1, 2, 3],
        "always_observed": [4, 5, 6, 7],
        "panels": [[0, 1], [2, 3]],
    }
    return X, y, roles
