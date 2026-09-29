import numpy as np
import pandas as pd
import pytest
import yaml

from data.clients import build_clients
from data.design import design_split, duplicate_groups
from data.inject import build_roles
from loop import scaler_for, shared_scaler


def _df_with_dups(n=2000, n_dup=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = X.sum(1)
    src = rng.choice(n, n_dup, replace=False)
    dst = rng.choice(np.setdiff1d(np.arange(n), src), n_dup, replace=False)
    df.iloc[dst] = df.iloc[src].to_numpy()
    return df


def test_relocation_is_identity_without_duplicates_and_moves_only_duplicates():
    n = 1500
    assert all(np.array_equal(a, b) for a, b in zip(design_split(n), design_split(n, groups=np.arange(n))))
    df = _df_with_dups()
    g = duplicate_groups(df)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    roles = build_roles(df)
    a, _ = build_clients(df, roles, cfg, 11)
    cfg["corrections"] = {"dup_groups": True}
    b, _ = build_clients(df, roles, cfg, 11)
    def where(clients, design):
        loc = np.full(len(df), "design", dtype=object)
        for c in clients:
            for f in ["train", "val", "test"]:
                loc[c["rows"][c[f]]] = f"{c['id']}:{f}"
        return loc
    la, lb = where(a, None), where(b, None)
    moved = np.flatnonzero(la != lb)
    _, counts = np.unique(g, return_counts=True)
    assert len(moved) > 0 and np.all(counts[g[moved]] > 1)              # only duplicate members move


def test_duplicate_groups_never_cross_any_boundary():
    df = _df_with_dups()
    g = duplicate_groups(df)
    assert len(np.unique(g)) == len(df) - 300
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cfg["corrections"] = {"dup_groups": True}
    roles = build_roles(df, g)
    clients, _ = build_clients(df, roles, cfg, 11)
    where = {}
    design, _ = design_split(len(df), groups=g)
    for r in design:
        where.setdefault(g[r], set()).add("design")
    for c in clients:
        for fold in ["train", "val", "test"]:
            for r in c["rows"][c[fold]]:
                where.setdefault(g[r], set()).add(f"{c['id']}:{fold}")
    assert all(len(v) == 1 for v in where.values())
    assert sum(len(c["rows"]) for c in clients) + len(design) == len(df)


def test_shared_scaler_is_frozen_on_design_rows_and_common_to_all_clients():
    df = _df_with_dups()
    g = duplicate_groups(df)
    roles = build_roles(df, g)
    cfg = {"corrections": {"dup_groups": True, "shared_scaler": True}}
    st = shared_scaler(df, roles, cfg)
    design, _ = design_split(len(df), groups=g)
    Xd = df[roles["features"]].to_numpy(float)[design]
    np.testing.assert_allclose(st.mu, np.median(Xd, axis=0))
    c = {"train": np.arange(10), "X": Xd[:10], "M": np.ones((10, 8)), "y": np.arange(10.0)}
    a = scaler_for(st, c)
    assert np.array_equal(a.mu, st.mu) and a.y_median == pytest.approx(4.5) and st.y_median != 4.5
    assert shared_scaler(df, roles, {}) is None


def test_random_maskable_assignment_is_target_independent():
    df = _df_with_dups()
    r1 = build_roles(df, maskable_mode="random")
    df2 = df.copy()
    df2["target"] = -df2["f0"] * 10                                 # completely different target
    r2 = build_roles(df2, maskable_mode="random")
    assert r1["maskable"] == r2["maskable"] and len(r1["maskable"]) == len(build_roles(df)["maskable"])


# ---------------------------------------------------------------- E5 (docs/PROTOCOL_E5.md)

def test_w_marginal_matches_hand_computation_and_is_directed():
    from scores import w_marginal
    r_i, r_j = np.array([0.5, 0.0, 0.25]), np.array([0.1, 0.4, 0.0])
    v, ok = w_marginal(r_i, r_j)
    assert ok and v == pytest.approx((0.5 * 0.9 + 0.0 * 0.6 + 0.25 * 1.0) / 0.75)
    assert w_marginal(r_j, r_i)[0] != pytest.approx(v)                  # directed
    assert w_marginal(np.zeros(3), r_j) == (None, False)                # receiver lacks nothing


def test_m0_preserves_per_feature_counts_and_destroys_association():
    from data.e5 import inject_panels_exact, permute_columns
    panels, maskable = [[0, 1], [2, 3]], [0, 1, 2, 3]
    rng = np.random.default_rng(0)
    m1 = inject_panels_exact(600, 5, panels, np.array([0.2, 0.9]), 0.05, rng)
    m0 = permute_columns(m1, maskable, np.random.default_rng(1))
    np.testing.assert_array_equal((m1 == 0).sum(0), (m0 == 0).sum(0))    # exact count match
    within = lambda M: np.corrcoef(1 - M[:, 0], 1 - M[:, 1])[0, 1]
    assert within(m1) > 0.6 and abs(within(m0)) < 0.15
    assert (m1[:, 4] == 1).all() and (m0[:, 4] == 1).all()               # never-masked untouched


def test_partition_column_is_target_free_and_follows_the_declared_rule():
    from data.e5 import partition_column
    df = _df_with_dups()
    roles = build_roles(df)
    a = partition_column(df, roles)
    df2 = df.copy()
    df2["target"] = -100 * df2["f0"]
    assert partition_column(df2, roles) == a          # same roles, different target -> same column
    assert a in roles["always_observed"]              # reliably observed (never masked)
    design, _ = design_split(len(df), groups=None)
    X = df[roles["features"]].to_numpy(float)[design]
    nuniq = {i: len(np.unique(X[:, i])) for i in roles["always_observed"]}
    assert nuniq[a] == max(nuniq.values())            # most distinct values, as declared
