"""Fetch the GRAPE-9 UCI suite from the official GRAPE repository (CLAUDE.md §2).

Writes raw/<name>.csv with feature columns f0..f{d-1} and a final `target` column.
Fails loudly on any native NaN: controlled injection needs complete data.
"""
import argparse
import io
import os
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/maxiaoba/GRAPE/master/uci/raw_data"
ALL = ["concrete", "energy", "housing", "kin8nm", "naval", "power", "protein", "wine", "yacht"]


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def _indices(text):
    return [int(tok) for tok in text.split()]


def fetch(name):
    root = f"{BASE}/{name}/data"
    data = np.loadtxt(io.StringIO(_get(f"{root}/data.txt")), dtype=float)
    feat = _indices(_get(f"{root}/index_features.txt"))
    targ = _indices(_get(f"{root}/index_target.txt"))
    if len(targ) != 1:
        raise ValueError(f"{name}: expected one target index, got {targ}")
    df = pd.DataFrame(data[:, feat], columns=[f"f{i}" for i in range(len(feat))])
    df["target"] = data[:, targ[0]]
    n_nan = int(df.isna().sum().sum())
    if n_nan:
        raise ValueError(f"{name}: {n_nan} native NaNs - injection needs complete data")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./raw")
    ap.add_argument("--datasets", nargs="+", default=["concrete", "wine", "kin8nm", "protein"], choices=ALL)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets:
        df = fetch(name)
        df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print(f"{name:10s} n={len(df):6d} d={df.shape[1] - 1:3d} NaNs=0")


if __name__ == "__main__":
    main()
