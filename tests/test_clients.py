import numpy as np
import pandas as pd
import pytest
import yaml

from data.clients import (MIN_TRAIN, asymmetric_receiver, build_clients, rare_panels,
                          size_check)
from data.inject import build_roles


def _setup(n=3000, mechanism="mcar"):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 8))
    X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]
    X[:, 3] = 0.8 * X[:, 2] + 0.6 * X[:, 3]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = X.sum(1) + 0.1 * rng.standard_normal(n)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cfg["injection"]["mechanism"] = mechanism
    roles = build_roles(df)
    return df, roles, cfg


def test_rare_panels_match_group_halves():
    assert rare_panels(2, 0) == [0] and rare_panels(2, 1) == [1]
    assert rare_panels(3, 0) == [0] and rare_panels(3, 1) == [2]     # middle panel is never rare
    assert rare_panels(4, 0) == [0, 1] and rare_panels(4, 1) == [2, 3]


@pytest.mark.parametrize("mechanism", ["mcar", "mar"])
def test_asymmetric_receiver_lowers_only_rare_panels_and_keeps_size(mechanism):
    df, roles, cfg = _setup(mechanism=mechanism)
    clients, _ = build_clients(df, roles, cfg, 11)
    for c in clients:
        rec = asymmetric_receiver(c, roles, cfg, 11)
        rare = rare_panels(len(c["profile"]), c["group"])
        shift = cfg["clients"]["receiver_p_rare"] - cfg["clients"]["p_rare"]
        np.testing.assert_allclose(rec["profile"][rare], c["profile"][rare] + shift)
        other = [k for k in range(len(c["profile"])) if k not in rare]
        np.testing.assert_array_equal(rec["profile"][other], c["profile"][other])
        for fold in ["train", "val", "test"]:
            np.testing.assert_array_equal(rec[fold], c[fold])        # same rows, same size
        rare_cols = [f for k in rare for f in roles["panels"][k]]
        other_cols = [f for k in other for f in roles["panels"][k]]
        assert rec["M"][:, rare_cols].mean() < c["M"][:, rare_cols].mean()   # observed less
        np.testing.assert_array_equal(rec["M"][:, other_cols], c["M"][:, other_cols])


def test_client_size_precondition():
    assert size_check(1030, 6, [0.6, 0.2, 0.2]) == ("FAIL", [88] * 6)
    status, sizes = size_check(1599, 6, [0.6, 0.2, 0.2])
    assert status == "OK" and min(sizes) >= MIN_TRAIN
    assert size_check(1330, 6, [0.6, 0.2, 0.2])[0] == "WARN"        # 106 rows: borderline


def test_build_clients_refuses_undersized_clients():
    df, roles, cfg = _setup(n=700)                                    # 4 clients x 82 rows
    with pytest.raises(ValueError, match="precondition failed"):
        build_clients(df, roles, cfg, 11)
