"""Panel-based missingness injection (CLAUDE.md §3).

Mask convention: M = 1 observed, M = 0 missing.

A panel is ordered for a row or not; if not, every feature in it is missing.
If ordered, each member is independently lost with prob `jitter`.
"""
import os

import numpy as np
import yaml

from data.design import DESIGN_SEED, decile_edges, design_split

MAR_SLOPE = 2.0        # b in sigmoid(a + b*z)
CD_SLOPE = 2.0         # logit shift per unit class_spread at the extreme outcome bin
CD_BINS = 4
MECHANISMS = ("cell", "mcar", "mar", "fd_mnar", "cd_mnar")


# ---------------------------------------------------------------- roles & panels

def _abs_corr(a, b):
    """|Pearson corr| between columns of a and b; a constant column correlates 0."""
    sa, sb = a.std(0), b.std(0)
    a = (a - a.mean(0)) / np.where(sa > 0, sa, 1.0)
    b = (b - b.mean(0)) / np.where(sb > 0, sb, 1.0)
    return np.abs(a.T @ b / len(a))


def form_panels(X):
    """Greedy panels of size 2-3 over the columns of X (indices into X)."""
    C = _abs_corr(X, X)
    d = C.shape[0]
    off = C[np.triu_indices(d, 1)]
    med = float(np.median(off)) if len(off) else 0.0
    free, panels = set(range(d)), []
    while len(free) >= 2:
        pairs = [(C[i, j], i, j) for i in free for j in free if i < j]
        _, i, j = max(pairs)
        panel = [i, j]
        rest = [(0.5 * (C[k, i] + C[k, j]), k) for k in free - {i, j}]
        if rest:
            score, k = max(rest)
            if score > med:
                panel.append(k)
        panels.append(sorted(panel))
        free -= set(panel)
    panels += [[k] for k in sorted(free)]
    return panels, med


def build_roles(df):
    """Column roles, panels and histogram bin edges, from the design split only.

    Features constant on the design rows carry no information: they are listed
    as `constant`, never masked and never used as characteristics or drivers.
    """
    feats = [c for c in df.columns if c != "target"]
    design, _ = design_split(len(df))
    X = df[feats].to_numpy(float)[design]
    y = df["target"].to_numpy(float)[design][:, None]
    constant = [i for i in range(len(feats)) if X[:, i].std() == 0]
    live = [i for i in range(len(feats)) if i not in constant]
    corr_y = _abs_corr(X, y)[:, 0]
    order = sorted(live, key=lambda i: (corr_y[i], i))
    n_mask = len(live) // 2
    maskable = sorted(order[:n_mask])
    always = sorted(order[n_mask:])
    local_panels, med = form_panels(X[:, maskable])
    panels = [[maskable[i] for i in p] for p in local_panels]
    return {
        "features": feats,
        "abs_corr_target": {feats[i]: round(float(corr_y[i]), 4) for i in live},
        "constant": constant,
        "maskable": maskable,
        "always_observed": always,
        "panels": panels,
        "panel_corr_median": round(med, 4),
        "bin_edges": {feats[i]: [float(e) for e in decile_edges(X[:, i])] for i in always},
        "design_seed": DESIGN_SEED,
        "n_design_rows": int(len(design)),
    }


def load_or_build_roles(name, df, cfg_dir="configs/roles"):
    path = os.path.join(cfg_dir, f"{name}.yaml")
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    roles = build_roles(df)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(roles, f, sort_keys=False)
    return roles


# ---------------------------------------------------------------- ordering probs

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate(logit_fn, target):
    """Find offset a so that mean(sigmoid(logit_fn(a))) == target (bisection)."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _sigmoid(logit_fn(mid)).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _standardise(X):
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


def _driver_scores(X_obs, n_panels, driver_overlap, driver_seed):
    """One standardised MAR driver per panel.

    Always-observed features are whitened, then n_panels + 1 orthonormal
    directions are drawn: u0 is shared, u_P is panel P's own. The panel driver
    is sqrt(o)*z_shared + sqrt(1-o)*z_own, so drivers of two panels correlate
    exactly `driver_overlap` in-sample: 0 -> uncorrelated, 1 -> identical.
    """
    Z = _standardise(X_obs)
    cov = np.cov(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
    keep = vals > 1e-8
    W = Z @ vecs[:, keep] / np.sqrt(vals[keep])          # whitened, identity cov
    m = W.shape[1]
    if n_panels + 1 > m:
        raise ValueError(f"MAR needs {n_panels + 1} driver directions, only {m} available")
    rng = np.random.default_rng(driver_seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, n_panels + 1)))
    S = W @ U
    S = (S - S.mean(0)) / S.std(0)
    o = float(driver_overlap)
    return np.sqrt(o) * S[:, [0]] + np.sqrt(1 - o) * S[:, 1:]


def ordering_probs(X, y, roles, p, mechanism, driver_overlap=0.0, class_spread=0.0,
                   direction="top", driver_seed=0):
    """q[r, P]: probability row r orders panel P. Realised mean per panel == p[P]."""
    panels = roles["panels"]
    n, K = len(X), len(panels)
    p = np.asarray(p, float)
    q = np.empty((n, K))
    if mechanism == "mcar":
        q[:] = p
    elif mechanism == "mar":
        Z = _driver_scores(X[:, roles["always_observed"]], K, driver_overlap, driver_seed)
        for k in range(K):
            a = _calibrate(lambda a: a + MAR_SLOPE * Z[:, k], p[k])
            q[:, k] = _sigmoid(a + MAR_SLOPE * Z[:, k])
    elif mechanism == "fd_mnar":
        Xs = _standardise(X)
        for k, P in enumerate(panels):
            v = Xs[:, P].mean(1)
            if direction == "bottom":
                v = -v
            n_skip = int(round((1 - p[k]) * n))
            rank = np.argsort(np.argsort(-v, kind="stable"), kind="stable")
            q[:, k] = (rank >= n_skip).astype(float)     # top n_skip rows skipped
    elif mechanism == "cd_mnar":
        edges = np.quantile(y, np.linspace(0, 1, CD_BINS + 1)[1:-1])
        b = np.searchsorted(edges, y, side="right")
        shift = class_spread * CD_SLOPE * np.linspace(-1, 1, CD_BINS)[b]
        if direction == "bottom":
            shift = -shift
        for k in range(K):
            base = np.log(p[k] / (1 - p[k]))
            a = _calibrate(lambda a: base + a + shift, p[k])
            q[:, k] = _sigmoid(base + a + shift)
    else:
        raise ValueError(f"unknown mechanism {mechanism}")
    return q


# ---------------------------------------------------------------- injection

def expected_feature_rates(roles, p, jitter):
    """Expected missing rate of every feature: 1 - p[P](1 - jitter) for f in P, 0 otherwise.
    Identical for all mechanisms, so conditions are rate-matched in expectation."""
    r = np.zeros(len(roles["features"]))
    for k, P in enumerate(roles["panels"]):
        r[P] = 1 - p[k] * (1 - jitter)
    return r


def inject(X, y, roles, p, mechanism, rng, jitter=0.05, **kw):
    """Return (M, ordered): M (n, d) mask, ordered (n, n_panels) bool.

    mechanism "cell" is the independence control (CLAUDE.md §3.6 (a)): every
    maskable cell is missing independently at its feature's expected rate, with
    no panel ordering at all (ordered is None)."""
    n, d = X.shape
    p = np.asarray(p, float)
    if mechanism == "cell":
        rate = expected_feature_rates(roles, p, jitter)
        M = np.ones((n, d), dtype=np.int8)
        for f in roles["maskable"]:
            M[:, f] = rng.random(n) >= rate[f]
        return M, None
    q = ordering_probs(X, y, roles, p, mechanism, **kw)
    ordered = rng.random(q.shape) < q
    M = np.ones((n, d), dtype=np.int8)
    for k, P in enumerate(roles["panels"]):
        for f in P:
            lost = rng.random(n) < jitter
            M[:, f] = ordered[:, k] & ~lost
    return M, ordered


def panel_probs_for_rate(n_panels, rate, jitter):
    """Uniform per-panel ordering prob giving `rate` missing among maskable cells."""
    p = (1 - rate) / (1 - jitter)
    if not 0 < p < 1:
        raise ValueError(f"rate {rate} with jitter {jitter} is infeasible")
    return np.full(n_panels, p)
