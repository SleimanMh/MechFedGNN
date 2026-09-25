import warnings

import numpy as np
import pandas as pd
import pytest

from data.clients import assign_rows_homogeneous, client_pool
from data.design import design_size, design_split, load_dataset
from data.inject import build_roles


def test_design_rows_never_appear_in_any_client():
    for n in [506, 1030, 1599, 11934]:
        design, pool = design_split(n)
        assert len(design) == design_size(n)
        assert np.array_equal(np.sort(np.concatenate([design, pool])), np.arange(n))
        for seed in [11, 23, 37]:
            clients = assign_rows_homogeneous(n, 6, np.random.default_rng(seed))
            used = np.concatenate(clients)
            assert len(np.intersect1d(used, design)) == 0
            assert len(used) == len(np.unique(used))                 # disjoint
            assert np.array_equal(np.sort(used), client_pool(n))     # whole pool used


def test_design_split_is_fixed_across_run_seeds():
    a, _ = design_split(1030)
    b, _ = design_split(1030)
    assert np.array_equal(a, b)


def test_roles_depend_only_on_design_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((1000, 8)), columns=[f"f{i}" for i in range(8)])
    df["target"] = df[["f0", "f1", "f2"]].sum(1) + 0.1 * rng.standard_normal(1000)
    _, pool = design_split(len(df))
    perturbed = df.copy()
    perturbed.loc[pool, :] = rng.standard_normal((len(pool), 9)) * 100
    assert build_roles(df) == build_roles(perturbed)


def test_design_size_floor_and_cap():
    assert design_size(506) == 150            # floor of 150 beats 10%
    assert design_size(1599) == 160           # 10% beats the floor
    assert design_size(300) == 90             # cap at 30% of n
    with pytest.warns(UserWarning, match="only 90 rows"):
        design_split(300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        design_split(1030)                    # >= 150 rows: no warning


def test_load_drops_exact_duplicate_columns(tmp_path):
    df = pd.DataFrame({"f0": [1.0, 2, 3], "f1": [4.0, 5, 6], "f2": [1.0, 2, 3], "target": [0.0, 1, 0]})
    df.to_csv(tmp_path / "toy.csv", index=False)
    out, dropped = load_dataset("toy", str(tmp_path), verbose=False)
    assert dropped == ["f2"] and list(out.columns) == ["f0", "f1", "target"]
