"""Design split and dataset loading.

The design split is a fixed set of rows held out of every client, used ONLY for
column roles, panel assignment and histogram bin edges, so those frozen design
choices never see a row that some client later trains, tunes or tests on.
"""
import os
import warnings

import numpy as np
import pandas as pd

DESIGN_SEED = 0
DESIGN_FRAC = 0.10
DESIGN_MIN = 150
DESIGN_CAP = 0.30


def design_size(n):
    """max(10% of n, 150), capped at 30% of n."""
    return int(min(max(round(DESIGN_FRAC * n), DESIGN_MIN), int(DESIGN_CAP * n)))


def design_split(n, seed=DESIGN_SEED):
    """Return (design_idx, pool_idx), sorted, disjoint, covering range(n)."""
    n_design = design_size(n)
    if n_design < DESIGN_MIN:
        warnings.warn(f"design split has only {n_design} rows (< {DESIGN_MIN}); "
                      "roles, panels and bin edges will be noisy")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:n_design]), np.sort(perm[n_design:])


def decile_edges(x):
    """Unique decile edges of x; fewer than 11 when x has ties."""
    return np.unique(np.quantile(x, np.linspace(0, 1, 11)))


def load_dataset(name, raw_dir="./raw", verbose=True):
    """Read raw/<name>.csv and drop exact-duplicate feature columns (keep first)."""
    df = pd.read_csv(os.path.join(raw_dir, f"{name}.csv"))
    feats = [c for c in df.columns if c != "target"]
    dup = df[feats].T.duplicated().to_numpy()
    dropped = [c for c, d in zip(feats, dup) if d]
    if dropped:
        df = df.drop(columns=dropped)
    if verbose:
        msg = f" (dropped exact duplicates: {dropped})" if dropped else ""
        print(f"[load] {name}: n={len(df)} d={len(feats) - len(dropped)} of {len(feats)}{msg}")
    return df, dropped
