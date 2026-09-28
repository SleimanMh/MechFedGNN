"""Design split and dataset loading.

The design split is a fixed set of rows held out of every client, used ONLY for
column roles, panel assignment and histogram bin edges, so those frozen design
choices never see a row that some client later trains, tunes or tests on.
"""
import os

import numpy as np
import pandas as pd

DESIGN_SEED = 0
DESIGN_FRAC = 0.10
DESIGN_MIN = 150


def design_size(n):
    """Flat 10% of n; the 150-row floor applies only when 10% is smaller."""
    if n <= DESIGN_MIN:
        raise ValueError(f"n={n} rows cannot spare a {DESIGN_MIN}-row design split")
    return int(max(round(DESIGN_FRAC * n), DESIGN_MIN))


def duplicate_groups(df):
    """Group id per row: rows with identical feature vector AND target share an id."""
    key = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return np.unique(key, return_inverse=True)[1]


def representatives(groups):
    """For every row, the lowest row index of its duplicate group."""
    groups = np.asarray(groups)
    rep = np.full(groups.max() + 1, len(groups))
    np.minimum.at(rep, groups, np.arange(len(groups)))
    return rep[groups]


def design_split(n, seed=DESIGN_SEED, groups=None):
    """Return (design_idx, pool_idx), sorted, disjoint, covering range(n).
    With `groups`, every duplicate group follows its representative (lowest row
    index), so no group crosses the boundary and only duplicate members move."""
    perm = np.random.default_rng(seed).permutation(n)
    n_design = design_size(n)
    in_design = np.zeros(n, dtype=bool)
    in_design[perm[:n_design]] = True
    if groups is not None:                       # a group goes where its representative went
        in_design = in_design[representatives(groups)]
    return np.flatnonzero(in_design), np.flatnonzero(~in_design)


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
