"""E1M matched-marginal construction (CLAUDE.md §10, E1M).

Group A masks with pair partition pi_A, group B with pi_B, over the same
maskable features and with the same ordering probability for every panel, so
per-feature rates are flat by design. Masks are injected PER FOLD with EXACT
counts, so every maskable feature in a fold has the same missing count.

Rows and splits per client are E1's (same RNG draws), so populations and
training conditions match E1 exactly.
"""
import itertools

import numpy as np

from data.clients import _split, assign_rows_homogeneous, client_profiles, dup_groups, relocate_duplicates
from data.design import design_split
from data.inject import _abs_corr, _calibrate, _driver_scores, _sigmoid

FOLDS = ("train", "val", "test")
CONDITIONS = {"a": "cell", "b": "mcar", "c": "mar"}


# ---------------------------------------------------------------- partitions

def _pairings(items):
    """All perfect pairings of an even-sized list, each as a sorted tuple of sorted pairs."""
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for k, other in enumerate(rest):
        for tail in _pairings(rest[:k] + rest[k + 1:]):
            yield tuple(sorted(((first, other),) + tail))


def partitions(df, roles, groups=None):
    """(pi_A, pi_B, singleton, quality) by the rule declared in CLAUDE.md before computation."""
    mk = list(roles["maskable"])
    design, _ = design_split(len(df), groups=groups)
    X = df[roles["features"]].to_numpy(float)[design]
    C = _abs_corr(X, X)
    free, pi_a = set(mk), []
    for f, g in sorted(itertools.combinations(mk, 2), key=lambda t: (-C[t[0], t[1]], t)):
        if f in free and g in free:
            pi_a.append((f, g))
            free -= {f, g}
    singleton = sorted(free)[0] if free else None
    pi_a = tuple(sorted(pi_a))
    mean_corr = lambda pi: float(np.mean([C[f, g] for f, g in pi]))
    target = mean_corr(pi_a)
    cands = [pi for pi in _pairings(sorted(f for f in mk if f != singleton))
             if not set(pi) & set(pi_a)]
    if not cands:
        raise ValueError("no pairing shares no pair with pi_A")
    pi_b = min(cands, key=lambda pi: (abs(mean_corr(pi) - target), pi))
    quality = {"mean_corr_A": target, "mean_corr_B": mean_corr(pi_b),
               "abs_diff": abs(mean_corr(pi_b) - target), "n_candidates": len(cands)}
    return pi_a, pi_b, singleton, quality


def panels_of(pi, singleton):
    return [list(p) for p in pi] + ([[singleton]] if singleton is not None else [])


# ---------------------------------------------------------------- balanced injection

def inject_balanced(X, panels, maskable, always, p, mechanism, rng, jitter, driver_overlap, driver_seed):
    """Exact counts: round(p n) ordered rows per panel, round(jitter * n_ordered)
    lost per member; cell masking removes the same count per feature."""
    n, d = X.shape
    M = np.ones((n, d), dtype=np.int8)
    n_ord = int(round(p * n))
    n_lost = int(round(jitter * n_ord))
    if mechanism == "cell":
        for f in maskable:
            M[rng.choice(n, n - n_ord + n_lost, replace=False), f] = 0
        return M
    if mechanism == "mar":
        Z = _driver_scores(X[:, always], len(panels), driver_overlap, driver_seed)
    for k, P in enumerate(panels):
        if mechanism == "mcar":
            ordered = rng.choice(n, n_ord, replace=False)
        elif mechanism == "mar":
            a = _calibrate(lambda a: a + 2.0 * Z[:, k], p)
            q = _sigmoid(a + 2.0 * Z[:, k])
            ordered = rng.choice(n, n_ord, replace=False, p=q / q.sum())
        else:
            raise ValueError(mechanism)
        on = np.zeros(n, dtype=bool)
        on[ordered] = True
        for f in P:
            col = on.copy()
            col[rng.choice(ordered, n_lost, replace=False)] = False
            M[:, f] = col
    return M


def build_matched_clients(df, roles, cfg, seed, condition, rate=0.3):
    """E1M clients for one seed: E1's rows and splits; group A masks with pi_A,
    group B with pi_B; exact counts per fold. Returns (clients, groups, info)."""
    cc, inj = cfg["clients"], cfg["injection"]
    X = df[roles["features"]].to_numpy(float)
    y = df["target"].to_numpy(float)
    rng = np.random.default_rng([seed, 0])
    _, groups = client_profiles(len(roles["panels"]), rng, cc["K"], cc["G"], cc["profile_noise"],
                                cc["p_rare"], cc["p_usual"])        # same draws as E1 -> same rows
    groups_dup = dup_groups(df, cfg)
    parts = assign_rows_homogeneous(len(X), cc["K"], rng)
    splits = [_split(len(rows), cc["split"], np.random.default_rng([seed, 3, k])) for k, rows in enumerate(parts)]
    if groups_dup is not None:
        parts, splits = relocate_duplicates(len(X), parts, splits, groups_dup)
    pi_a, pi_b, singleton, quality = partitions(df, roles, groups_dup)
    jitter = inj["jitter"]
    p = (1 - rate) / (1 - jitter)
    clients = []
    for k, rows in enumerate(parts):
        split = splits[k]
        panels = panels_of(pi_a if groups[k] == 0 else pi_b, singleton)
        M = np.ones((len(rows), X.shape[1]), dtype=np.int8)
        for fi, fold in enumerate(FOLDS):
            idx = split[fold]
            M[idx] = inject_balanced(X[rows][idx], panels, roles["maskable"], roles["always_observed"], p,
                                     CONDITIONS[condition], np.random.default_rng([seed, 5, k, fi]),
                                     jitter, inj["driver_overlap"], seed)
        clients.append({"id": f"c{k}", "k": k, "rows": rows, "X": X[rows], "y": y[rows], "M": M,
                        "group": int(groups[k]), "panels": panels, **split})
    info = {"pi_A": pi_a, "pi_B": pi_b, "singleton": singleton, "quality": quality,
            "p": p, "expected_rate": 1 - p * (1 - jitter)}
    return clients, groups, info
