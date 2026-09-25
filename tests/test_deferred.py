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
    rows = run_seed(df, build_roles(df), cfg, seed, split="val")["headroom"]
    return [r["verdict"] for r in rows], [r["rel_headroom"] for r in rows]


def test_headroom_ok_for_small_receiver():
    df = _synthetic_df(6000)
    v, rel = _verdicts(df, _cfg(**{"clients.receiver_fraction": 0.05}))
    assert all(x == "HEADROOM OK" for x in v), (v, rel)


def test_no_headroom_when_receiver_gets_full_sized_sample():
    # A linear target that ~1600 local rows already pin down, and enough local
    # steps to converge: with the default 300 the remaining gap is optimisation
    # (the oracle gets K x the steps), not data.
    df = _synthetic_df(12000, noise=1.0, linear=True)
    v, rel = _verdicts(df, _cfg(**{"clients.receiver_fraction": 1.0, "model.local_steps": 1500}))
    assert all(x == "NO HEADROOM" for x in v), (v, rel)


def test_determinism_same_seed_identical_csvs(tmp_path):
    df = _synthetic_df(900)
    cfg = _cfg(**{"model.local_steps": 40, "model.adapt_budget": 5})
    roles = build_roles(df)
    for run in ["a", "b"]:
        write_csvs(tmp_path / run, [run_seed(df, roles, copy.deepcopy(cfg), 11, split="val")])
    for f in ["metrics.csv", "scores.csv"]:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    other = tmp_path / "c"
    write_csvs(other, [run_seed(df, roles, copy.deepcopy(cfg), 23, split="val")])
    assert (other / "metrics.csv").read_bytes() != (tmp_path / "a" / "metrics.csv").read_bytes()
