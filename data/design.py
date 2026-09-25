"""Design split: a fixed fraction of rows held out of every client.

Used ONLY for column roles, panel assignment and histogram bin edges, so those
frozen design choices never see a row that some client later trains or tests on.
"""
import numpy as np

DESIGN_SEED = 0
DESIGN_FRAC = 0.10


def design_split(n, frac=DESIGN_FRAC, seed=DESIGN_SEED):
    """Return (design_idx, pool_idx), sorted, disjoint, covering range(n)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_design = int(round(frac * n))
    return np.sort(perm[:n_design]), np.sort(perm[n_design:])


def decile_edges(x):
    """Unique decile edges of x; fewer than 11 when x has ties."""
    return np.unique(np.quantile(x, np.linspace(0, 1, 11)))
