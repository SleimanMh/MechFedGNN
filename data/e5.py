"""E5 population-compatibility construction (docs/PROTOCOL_E5.md).

Factorial: P0/P1 (population) x M0/M1 (missingness).

P1 uses a TARGET-FREE partition characteristic, declared in the protocol: the
always-observed feature with the most distinct values on the design rows.
`data.clients.partition_col` (highest |corr| with the target) is deliberately
NOT used here.

M1 is value-independent panel masking with exact per-fold counts; M0 permutes
each column of M1 independently, so per-feature counts are identical by
construction and only the joint structure differs.
"""
import numpy as np

from data.clients import client_profiles, dup_groups, rare_panels
from data.design import design_split, representatives

FOLDS = ("train", "val", "test")


# ---------------------------------------------------------------- population

def partition_column(df, roles, groups=None):
    """Declared rule: among always_observed, the feature with the most distinct
    values on the DESIGN rows; ties -> lowest column index. Target-free."""
    design, _ = design_split(len(df), groups=groups)
    X = df[roles["features"]].to_numpy(float)[design]
    return min(roles["always_observed"], key=lambda i: (-len(np.unique(X[:, i])), i))


def _group_rows(pool, groups):
    """{group id: array of pool rows}, and each group's representative row."""
    reps = representatives(groups)
    out = {}
    for r in pool:
        out.setdefault(reps[r], []).append(r)
    return {g: np.array(v) for g, v in out.items()}


def assign_population(X, roles, cfg, rng, groups, pcol):
    """P1: client k draws a frac_low[k] share of its rows from the below-median
    half of `pcol`. Whole duplicate groups only; all pool rows assigned; client
    sizes kept as close as feasible by filling the largest remaining deficit."""
    cc = cfg["clients"]
    K, frac_low = cc["K"], cc["frac_low"]
    _, pool = design_split(len(X), groups=groups)
    gr = _group_rows(pool, groups)
    med = np.median(X[pool, pcol])
    low = [g for g, rows in gr.items() if X[rows[0], pcol] <= med]
    high = [g for g in gr if g not in set(low)]
    rng.shuffle(low)
    rng.shuffle(high)
    n_c = len(pool) // K
    quota = {"low": [int(round(f * n_c)) for f in frac_low],
             "high": [n_c - int(round(f * n_c)) for f in frac_low]}
    got = {"low": [0] * K, "high": [0] * K}
    parts = [[] for _ in range(K)]
    for half, gs in [("low", low), ("high", high)]:
        for g in gs:                                   # fill the largest deficit first
            k = int(np.argmax([quota[half][j] - got[half][j] for j in range(K)]))
            parts[k].append(gr[g])
            got[half][k] += len(gr[g])
    return [np.sort(np.concatenate(p)) for p in parts]


# ---------------------------------------------------------------- missingness

def panel_probs(n_panels, group, cfg):
    """Group profile: rarely-ordered panels at p_rare, the rest at p_usual."""
    cc = cfg["clients"]
    p = np.full(n_panels, cc["p_usual"], float)
    p[rare_panels(n_panels, group)] = cc["p_rare"]
    return p


def inject_panels_exact(n, d, panels, p_vec, jitter, rng):
    """M1: exactly round(p*n) rows order each panel, and exactly
    round(jitter*n_ordered) of those lose each member feature."""
    M = np.ones((n, d), dtype=np.int8)
    for k, P in enumerate(panels):
        n_ord = int(round(p_vec[k] * n))
        ordered = rng.choice(n, n_ord, replace=False)
        on = np.zeros(n, dtype=bool)
        on[ordered] = True
        n_lost = int(round(jitter * n_ord))
        for f in P:
            col = on.copy()
            if n_lost:
                col[rng.choice(ordered, n_lost, replace=False)] = False
            M[:, f] = col
    return M


def permute_columns(M, maskable, rng):
    """M0: permute each maskable column independently. Per-feature counts are
    preserved exactly; row-level association is destroyed."""
    out = M.copy()
    for f in maskable:
        out[:, f] = M[rng.permutation(len(M)), f]
    return out


# ---------------------------------------------------------------- clients

def build_e5_clients(df, roles, cfg, seed):
    """Clients for one (condition, seed). cfg['e5'] = {'P': 0|1, 'M': 0|1}.
    Returns (clients, latent_groups, info)."""
    e5, cc, inj = cfg["e5"], cfg["clients"], cfg["injection"]
    X = df[roles["features"]].to_numpy(float)
    y = df["target"].to_numpy(float)
    d = X.shape[1]
    groups = dup_groups(df, cfg)
    if groups is None:
        raise ValueError("E5 requires the §16 duplicate correction (corrections.dup_groups)")
    rng = np.random.default_rng([seed, 0])
    _, latent = client_profiles(len(roles["panels"]), rng, cc["K"], cc["G"],
                                cc["profile_noise"], cc["p_rare"], cc["p_usual"])
    pcol = partition_column(df, roles, groups)
    if e5["P"] == 0:                                   # E1's corrected random path
        from data.clients import assign_rows_homogeneous, relocate_duplicates, _split
        parts = assign_rows_homogeneous(len(X), cc["K"], rng)
        splits = [_split(len(r), cc["split"], np.random.default_rng([seed, 3, k]))
                  for k, r in enumerate(parts)]
        parts, splits = relocate_duplicates(len(X), parts, splits, groups)
    else:
        parts = assign_population(X, roles, cfg, rng, groups, pcol)
        splits = []
        for k, rows in enumerate(parts):               # split whole groups within the client
            srng = np.random.default_rng([seed, 3, k])
            gr = _group_rows(rows, groups)
            gids = list(gr)
            srng.shuffle(gids)
            a = int(round(cc["split"][0] * len(rows)))
            b = a + int(round(cc["split"][1] * len(rows)))
            pos, cur = {}, 0
            for g in gids:
                fold = 0 if cur < a else (1 if cur < b else 2)
                pos[g] = fold
                cur += len(gr[g])
            local = {r: i for i, r in enumerate(rows)}
            splits.append({f: np.sort([local[r] for g, fo in pos.items() if fo == fi
                                       for r in gr[g]]) for fi, f in enumerate(FOLDS)})
    clients = []
    for k, rows in enumerate(parts):
        M = np.ones((len(rows), d), dtype=np.int8)
        p_vec = panel_probs(len(roles["panels"]), int(latent[k]), cfg)
        for fi, fold in enumerate(FOLDS):
            idx = splits[k][fold]
            mrng = np.random.default_rng([seed, 5, k, fi])
            m1 = inject_panels_exact(len(idx), d, roles["panels"], p_vec, inj["jitter"], mrng)
            M[idx] = m1 if e5["M"] == 1 else permute_columns(m1, roles["maskable"],
                                                             np.random.default_rng([seed, 6, k, fi]))
        clients.append({"id": f"c{k}", "k": k, "rows": rows, "X": X[rows], "y": y[rows], "M": M,
                        "profile": p_vec, "group": int(latent[k]), **splits[k]})
    info = {"partition_col": pcol, "partition_name": roles["features"][pcol],
            "frac_low": cc["frac_low"] if e5["P"] == 1 else None,
            "condition": f"P{e5['P']}M{e5['M']}"}
    return clients, latent, info
