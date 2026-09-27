"""§13 tests that need the Stage 2 pipeline: headroom states and determinism."""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from data.inject import build_roles
from loop import run_seed
from report import write_csvs


def _synthetic_df(n, seed=0, noise=0.1, linear=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    signal = X.sum(1) if linear else (np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]
                                      + np.abs(X[:, 3]) + X[:, 4:].sum(1))
    y = signal + noise * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


def _cfg(**over):
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cfg["clients"]["K"], cfg["clients"]["G"] = 4, 2
    cfg["model"]["adapt_budget"] = 0
    for path, v in over.items():
        sec, key = path.split(".")
        cfg[sec][key] = v
    return cfg


def _verdicts(df, cfg, seed=11):
    rows = run_seed(df, build_roles(df), cfg, seed, folds=("val",))["headroom"]
    return [r["verdict"] for r in rows], [r["rel_headroom"] for r in rows]


def test_headroom_ok_when_pooling_adds_information():
    # ~125 training rows per client on a nonlinear, low-noise target, and no
    # latent missingness groups (every panel usually ordered, MCAR): the pooled
    # rows (4x) carry information a local model cannot get. Under the default
    # group structure pooling can instead HURT (negative headroom) - that is a
    # property of the design, not of the detector, so it is not used here.
    df = _synthetic_df(1000)
    v, rel = _verdicts(df, _cfg(**{"injection.mechanism": "mcar", "clients.p_rare": 0.9,
                                   "clients.receiver_p_rare": 0.9}))
    assert all(x == "HEADROOM OK" for x in v), (v, rel)


def test_no_headroom_when_receiver_sample_is_abundant():
    # ~1600 training rows per client on a linear target: the receiver's own
    # sample already pins the function down. Both references are early-stopped
    # on the same validation fold, so no step-count advantage remains.
    df = _synthetic_df(12000, noise=1.0, linear=True)
    v, rel = _verdicts(df, _cfg())
    assert all(x == "NO HEADROOM" for x in v), (v, rel)


def test_determinism_same_seed_identical_csvs(tmp_path):
    df = _synthetic_df(900)
    cfg = _cfg(**{"model.local_steps": 40, "model.adapt_budget": 5})
    roles = build_roles(df)
    for run in ["a", "b"]:
        write_csvs(tmp_path / run, [run_seed(df, roles, copy.deepcopy(cfg), 11, folds=("val",))])
    for f in ["metrics.csv", "scores.csv"]:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    other = tmp_path / "c"
    write_csvs(other, [run_seed(df, roles, copy.deepcopy(cfg), 23, folds=("val",))])
    assert (other / "metrics.csv").read_bytes() != (tmp_path / "a" / "metrics.csv").read_bytes()
