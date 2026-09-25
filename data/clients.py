"""Client construction (CLAUDE.md §4).

Missingness structure (latent group profile) and population (row assignment)
vary independently. Rows come only from the non-design pool and are disjoint
across clients. Each client: split 60/20/20 first, mask injected with its own
profile, summaries later taken from training rows only.
"""
import numpy as np

from data.design import design_split
from data.inject import inject

P_RARE = 0.2
P_USUAL = 0.9
PROFILE_NOISE = 0.05
MIN_TRAIN = 100        # precondition: every client has at least this many training rows
BORDERLINE_TRAIN = 120 # below this, warn


def group_profiles(n_panels, p_rare=P_RARE, p_usual=P_USUAL):
    """(G=2, n_panels) ordering probabilities.

    Group A rarely orders the first half of the ordered panel list, group B the
    second half. With an odd count the middle panel stays usually-ordered for
    both groups - a within-design control that no score should favour.
    """
    half = n_panels // 2
    a = np.full(n_panels, p_usual)
    b = np.full(n_panels, p_usual)
    a[:half] = p_rare
    b[n_panels - half:] = p_rare
    return np.stack([a, b])


def client_profiles(n_panels, rng, K=6, G=2, noise=PROFILE_NOISE, p_rare=P_RARE, p_usual=P_USUAL):
    """Per-client profiles = group profile + U(-noise, noise), clipped to [0.05, 0.95].

    Returns (profiles (K, n_panels), groups (K,)). Group labels are ground truth
    for evaluation only - never an input to any score.
    """
    groups = np.repeat(np.arange(G), K // G)
    base = group_profiles(n_panels, p_rare, p_usual)[groups]
    prof = np.clip(base + rng.uniform(-noise, noise, base.shape), 0.05, 0.95)
    return prof, groups


def rare_panels(n_panels, group):
    """Indices of the panels the group habitually skips (see group_profiles)."""
    half = n_panels // 2
    return list(range(half)) if group == 0 else list(range(n_panels - half, n_panels))


def client_train_sizes(n, K, split):
    """Training-fold size of each client under homogeneous assignment."""
    parts = np.array_split(np.arange(len(client_pool(n))), K)
    return [int(round(split[0] * len(p))) for p in parts]


def size_check(n, K, split):
    """('OK' | 'WARN' | 'FAIL', sizes). FAIL if any client trains on < MIN_TRAIN rows."""
    sizes = client_train_sizes(n, K, split)
    status = "FAIL" if min(sizes) < MIN_TRAIN else "WARN" if min(sizes) < BORDERLINE_TRAIN else "OK"
    return status, sizes


def client_pool(n):
    """Rows eligible for clients: everything outside the design split."""
    return design_split(n)[1]


def assign_rows_homogeneous(n, K, rng):
    """Shuffle the pool and split it into K disjoint, near-equal parts."""
    pool = rng.permutation(client_pool(n))
    return [np.sort(part) for part in np.array_split(pool, K)]


def partition_col(roles):
    """Always-observed feature with the highest |corr| with the target."""
    feats = roles["features"]
    return max(roles["always_observed"], key=lambda i: roles["abs_corr_target"][feats[i]])


def assign_rows_stratified(X, frac_low, rng, roles):
    """Split the pool at the partition column's median into LOW / HIGH; client k
    draws a frac_low[k] share of its rows from LOW. Disjoint, equal sizes."""
    pool = client_pool(len(X))
    col = X[pool, partition_col(roles)]
    low = rng.permutation(pool[col <= np.median(col)])
    high = rng.permutation(pool[col > np.median(col)])
    K = len(frac_low)
    n_c = len(pool) // K
    while (sum(round(f * n_c) for f in frac_low) > len(low)
           or sum(n_c - round(f * n_c) for f in frac_low) > len(high)):
        n_c -= 1
    out, i_lo, i_hi = [], 0, 0
    for f in frac_low:
        n_lo = round(f * n_c)
        rows = np.concatenate([low[i_lo:i_lo + n_lo], high[i_hi:i_hi + n_c - n_lo]])
        i_lo, i_hi = i_lo + n_lo, i_hi + n_c - n_lo
        out.append(np.sort(rows))
    return out


def _split(n, fracs, rng):
    perm = rng.permutation(n)
    a = int(round(fracs[0] * n))
    b = a + int(round(fracs[1] * n))
    return {"train": np.sort(perm[:a]), "val": np.sort(perm[a:b]), "test": np.sort(perm[b:])}


def build_clients(df, roles, cfg, seed):
    """All K clients for one run seed. Returns (clients, groups)."""
    cc, inj = cfg["clients"], cfg["injection"]
    X = df[roles["features"]].to_numpy(float)
    y = df["target"].to_numpy(float)
    rng = np.random.default_rng([seed, 0])
    prof, groups = client_profiles(len(roles["panels"]), rng, cc["K"], cc["G"],
                                   cc["profile_noise"], cc["p_rare"], cc["p_usual"])
    if cc["population"] == "population_homogeneous":
        parts = assign_rows_homogeneous(len(X), cc["K"], rng)
    elif cc["population"] == "population_stratified":
        parts = assign_rows_stratified(X, cc["frac_low"], rng, roles)
    else:
        raise ValueError(cc["population"])
    clients = []
    for k, rows in enumerate(parts):
        M = _inject_client(X[rows], y[rows], roles, prof[k], inj, seed, k)
        split = _split(len(rows), cc["split"], np.random.default_rng([seed, 3, k]))
        clients.append({"id": f"c{k}", "k": k, "rows": rows, "X": X[rows], "y": y[rows], "M": M,
                        "profile": prof[k], "group": int(groups[k]), **split})
    small = [c["id"] for c in clients if len(c["train"]) < MIN_TRAIN]
    if small:
        raise ValueError(f"precondition failed: clients {small} have < {MIN_TRAIN} training rows "
                         f"({[len(c['train']) for c in clients]})")
    return clients, groups


def _inject_client(X, y, roles, profile, inj, seed, k):
    """Deterministic per (seed, k): re-injecting with another profile reuses the
    same random draws, so the receiver's asymmetric mask differs only through
    the ordering probabilities (common random numbers)."""
    return inject(X, y, roles, profile, inj["mechanism"], np.random.default_rng([seed, 1, k]),
                  jitter=inj["jitter"], driver_overlap=inj["driver_overlap"],
                  class_spread=inj["class_spread"], direction=inj["direction"],
                  driver_seed=seed)[0]


def asymmetric_receiver(client, roles, cfg, seed):
    """Copy of `client` as receiver: its ordering probability on the panels its
    group skips is lowered by (receiver_p_rare - p_rare), e.g. 0.20 -> 0.15, and
    the mask is re-injected. Size, rows and splits are unchanged."""
    cc = cfg["clients"]
    prof = client["profile"].copy()
    idx = rare_panels(len(prof), client["group"])
    prof[idx] = np.clip(prof[idx] + cc["receiver_p_rare"] - cc["p_rare"], 0.01, 0.95)
    M = _inject_client(client["X"], client["y"], roles, prof, cfg["injection"], seed, client["k"])
    return {**client, "M": M, "profile": prof}
