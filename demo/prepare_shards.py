"""Prepare one data shard per client for the separate-process demonstration.

Each client process is given ONLY its own file. The server is given none of
them - it never sees features, labels or per-row masks.

Run:  python -m demo.prepare_shards [--out demo/_shards] [--clients 3]
"""
import argparse
import json
import os

import numpy as np

from mechfedgnn.client_runtime import preprocessing_id, save_client_shard, shared_coordinates
from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.protocol import schema_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("demo", "_shards"))
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-client-scaling", action="store_true",
                    help="each client standardises on its own training fold. Their parameters "
                         "are then NOT in a common space; the coordinator will refuse to mix "
                         "them with shared-coordinate clients")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = 6
    os.makedirs(args.out, exist_ok=True)
    # deliberately UNEQUAL shard sizes, so sample-size weighting is non-uniform
    sizes = [260, 200, 140][:args.clients] + [180] * max(0, args.clients - 3)
    ids = [f"c{i}" for i in range(args.clients)]
    pool_X, pool_M, pool_y = [], [], []
    for cid, n in zip(ids, sizes):
        X = rng.standard_normal((n, d))
        y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.2 * rng.standard_normal(n)
        M = np.ones((n, d), dtype=np.int8)
        # each client is incomplete in its own way (no injector involved here)
        for f in (2, 3, 4, 5):
            M[rng.random(n) < (0.10 + 0.08 * (int(cid[1:]) % 3)), f] = 0
        cut1, cut2 = int(0.6 * n), int(0.8 * n)
        perm = rng.permutation(n)
        folds = {"train": np.sort(perm[:cut1]), "val": np.sort(perm[cut1:cut2]),
                 "test": np.sort(perm[cut2:])}
        roles = {"always_observed": [0, 1], "maskable": [2, 3, 4, 5]}
        save_client_shard(os.path.join(args.out, f"{cid}.npz"), X, M, y, folds, roles, cid)
        pool_X.append(X[folds["train"]]); pool_M.append(M[folds["train"]])
        pool_y.append(y[folds["train"]])
        print(f"  {cid}: n={n} train={len(folds['train'])} shard={cid}.npz")

    # ONE set of standardisation coordinates for everybody, mirroring the §16
    # correction (loop.shared_scaler). Fitted once here - a simulation
    # assumption, exactly as in the research protocol - and frozen. Without it
    # each client's parameters live in its own coordinates and averaging them
    # means nothing.
    shared = None if args.per_client_scaling else shared_coordinates(
        np.concatenate(pool_X), np.concatenate(pool_M), np.concatenate(pool_y))

    learner = MaskAwareMLP(hidden=(16, 8))
    state = learner.initial_state(d, seed=0)
    meta = {"clients": ids, "n_features": d, "hidden": [16, 8],
            "schema_id": schema_id(state),
            "shared_scale": shared, "preprocessing_id": preprocessing_id(shared),
            "feature_names": [f"f{i}" for i in range(d)],
            "bin_edges": {f"f{i}": list(np.linspace(-3, 3, 11)) for i in (0, 1)}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}/meta.json  schema={meta['schema_id']}  "
          f"preprocessing={meta['preprocessing_id']}")


if __name__ == "__main__":
    main()
