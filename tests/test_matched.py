import numpy as np
import pandas as pd
import pytest
import yaml

from data.inject import build_roles
from data.matched import FOLDS, build_matched_clients, partitions


def _df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 9))
    X[:, 1] = 0.7 * X[:, 0] + 0.7 * X[:, 1]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(9)])
    df["target"] = X[:, 5:].sum(1) + 0.1 * rng.standard_normal(n)
    return df


def test_partitions_are_pairings_sharing_no_pair():
    df = _df()
    roles = build_roles(df)
    pi_a, pi_b, single, q = partitions(df, roles)
    assert not set(pi_a) & set(pi_b)
    for pi in (pi_a, pi_b):
        used = sorted(f for pair in pi for f in pair) + ([single] if single is not None else [])
        assert sorted(used) == sorted(roles["maskable"])
    assert q["abs_diff"] == pytest.approx(abs(q["mean_corr_A"] - q["mean_corr_B"]))


@pytest.mark.parametrize("cond", ["a", "b", "c"])
def test_exact_counts_equal_across_features_within_every_fold(cond):
    df = _df()
    roles = build_roles(df)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    clients, groups, info = build_matched_clients(df, roles, cfg, 11, cond)
    for c in clients:
        for fold in FOLDS:
            miss = (c["M"][c[fold]][:, roles["maskable"]] == 0).sum(0)
            assert len(set(miss.tolist())) == 1                        # identical count per feature
        assert (c["M"][:, roles["always_observed"]] == 1).all()
    assert [c["panels"] for c in clients if c["group"] == 0][0] != [c["panels"] for c in clients if c["group"] == 1][0]


def test_run_seed_with_matched_builder_logs_geometry_and_groups(tmp_path):
    import copy

    from loop import run_seed
    from report import write_run

    df = _df(1200)
    roles = build_roles(df)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cfg["model"]["local_steps"], cfg["model"]["adapt_budget"] = 20, 3
    cfg["seeds"] = [11, 23]
    builder = lambda seed: build_matched_clients(df, roles, cfg, seed, "b")[:2]
    outs = [run_seed(df, roles, copy.deepcopy(cfg), s, folds=("val", "test"), headroom=False, builder=builder)
            for s in cfg["seeds"]]
    geo = pd.DataFrame(outs[0]["geometry"])
    assert set(geo.kind) == {"client-client", "client-theta0", "arm", "single"}
    fed = geo[(geo.kind == "arm") & (geo.a == "fedavg")]
    uni = geo[(geo.kind == "arm") & (geo.a == "uniform-donor")]
    assert fed.to_centroid.max() < 0.05 * uni.to_centroid.min()   # near-equal sizes: fedavg ~ the centroid
    sc = pd.DataFrame(outs[0]["scores"])
    assert sc.groupby(["receiver"])["same_group"].sum().eq(8).all()   # 1 same-group donor x 8 scores
    cfg["e1m"] = {"condition": "b", "mechanism": "mcar", "pi_A": ["x"], "pi_B": ["y"], "singleton": None,
                  "p": 0.7, "corr_diff": 0.0}
    out = write_run("t", cfg, "syn", df, roles, [], outs, root=str(tmp_path))
    assert "E1M condition (b)" in open(f"{out}/REPORT.md", encoding="utf-8").read()
