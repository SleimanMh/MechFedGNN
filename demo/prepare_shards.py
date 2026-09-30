"""Prepare one data shard per client for the separate-process demonstration.

Each client process is given ONLY its own file. The server is given none of
them - it never sees features, labels or per-row masks.

Run:  python -m demo.prepare_shards [--out demo/_shards] [--clients 3]
"""
import argparse
import json
import os

import numpy as np

from mechfedgnn.client_runtime import save_client_shard
from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.protocol import schema_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("demo", "_shards"))
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = 6
    os.makedirs(args.out, exist_ok=True)
    # deliberately UNEQUAL shard sizes, so sample-size weighting is non-uniform
    sizes = [260, 200, 140][:args.clients] + [180] * max(0, args.clients - 3)
    ids = [f"c{i}" for i in range(args.clients)]
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
        print(f"  {cid}: n={n} train={len(folds['train'])} shard={cid}.npz")

    learner = MaskAwareMLP(hidden=(16, 8))
    state = learner.initial_state(d, seed=0)
    meta = {"clients": ids, "n_features": d, "hidden": [16, 8],
            "schema_id": schema_id(state),
            "feature_names": [f"f{i}" for i in range(d)],
            "bin_edges": {f"f{i}": list(np.linspace(-3, 3, 11)) for i in (0, 1)}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}/meta.json  schema={meta['schema_id']}")


if __name__ == "__main__":
    main()
