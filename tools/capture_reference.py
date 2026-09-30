"""Capture golden reference cases from the CURRENT implementation (Milestone A).

Run this BEFORE extracting interfaces. The saved fixtures are what the
refactored code must reproduce. They use a synthetic dataset on purpose: `raw/`
is gitignored, so a fresh clone must be able to run the equivalence tests.

Captured, per builder (E1 / E1M / E5):
  * client row indices and train/val/test fold indices;
  * masks (exact);
  * signatures r, H, J, C and population histograms (exact);
  * every pairwise score, including its None decisions;
  * arm weights, gamma and fallback reason - for equal AND NON-UNIFORM sizes;
  * aggregation output for a deliberately non-uniform weight vector;
  * a small end-to-end run_seed metrics table.

Run:  python -m tools.capture_reference
"""
import copy
import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

from data.clients import build_clients
from data.e5 import build_e5_clients
from data.inject import build_roles
from data.matched import build_matched_clients
from kernel import aggregate, donor_weights
from loop import ARMS, E5_ARMS, arm_weights, pair_scores, run_seed, summaries

OUT = os.path.join("tests", "golden")
SEED = 11


def synthetic_df(n=1200, d=8, seed=0):
    """Small, duplicate-bearing, deterministic. Kept tiny so fixtures stay small."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).round(4)
    X[:, 1] = (0.8 * X[:, 0] + 0.6 * X[:, 1]).round(4)
    X[:, 3] = (0.8 * X[:, 2] + 0.6 * X[:, 3]).round(4)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["target"] = (X.sum(1) + 0.1 * rng.standard_normal(n)).round(4)
    dup = rng.choice(n, 60, replace=False)
    dst = rng.choice(np.setdiff1d(np.arange(n), dup), 60, replace=False)
    df.iloc[dst] = df.iloc[dup].to_numpy()
    return df


def base_cfg():
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cfg["corrections"] = {"dup_groups": True, "shared_scaler": True, "maskable": "corr"}
    cfg["model"] = {**cfg["model"], "local_steps": 12, "adapt_budget": 3}
    cfg["seeds"] = [SEED]
    return cfg


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def capture_builder(name, clients, groups, roles, cfg, arms):
    """Everything derivable from a built client list, without training."""
    rec = {"clients": [], "signatures": {}, "scores": {}, "weights": {}}
    for c in clients:
        rec["clients"].append({
            "id": c["id"], "group": int(c["group"]), "n_rows": len(c["rows"]),
            "rows_sha": hashlib.sha256(np.ascontiguousarray(c["rows"]).tobytes()).hexdigest()[:16],
            "train": c["train"].tolist(), "val": c["val"].tolist(), "test": c["test"].tolist(),
            "mask_sha": hashlib.sha256(np.ascontiguousarray(c["M"]).tobytes()).hexdigest()[:16],
            "mask_col_missing": (c["M"] == 0).sum(0).tolist(),
        })
    summ = [summaries(c, roles) for c in clients]
    for k, s in enumerate(summ):
        rec["signatures"][clients[k]["id"]] = {
            "n_train": int(s["n_train"]),
            # full precision: these are deterministic pure functions of the mask,
            # so the equivalence test compares them exactly
            "r": s["sig"]["r"].tolist(),
            "H": s["sig"]["H"].tolist(),
            "J": s["sig"]["J"].tolist(),
            "C": s["sig"]["C"].tolist(),
            "hist": {kk: vv.tolist() for kk, vv in s["hist"].items()},
        }
    lam = cfg["aggregation"]["lambda_pop"]
    K = len(clients)
    for i in range(K):
        sc = {j: pair_scores(summ[i], summ[j], lam) for j in range(K) if j != i}
        rec["scores"][clients[i]["id"]] = {
            clients[j]["id"]: {k: (None if v is None else float(v)) if not isinstance(v, str) else v
                               for k, v in sc[j].items()} for j in sc}
        # equal sizes AND a deliberately non-uniform size vector
        for tag, sizes in [("equal", np.full(K, 1.0 / K)),
                           ("nonuniform", np.array([0.10, 0.20, 0.30, 0.40])[:K])]:
            p = sizes / sizes.sum()
            for arm in arms:
                w, gamma, fb = arm_weights(arm, i, sc, p, cfg["aggregation"])
                rec["weights"][f"{clients[i]['id']}|{tag}|{arm}"] = {
                    "w": None if w is None else w.tolist(),
                    "gamma": float(gamma), "fallback": fb}
    return rec


def capture_aggregation():
    """Aggregator with a NON-UNIFORM weight vector, so an equal-weight bug cannot hide."""
    rng = np.random.default_rng(7)
    thetas = [{"a": rng.standard_normal(6).round(6), "b": rng.standard_normal((2, 3)).round(6)}
              for _ in range(4)]
    w = np.array([0.0, 0.5, 0.3, 0.2])
    out = aggregate(thetas, 0, w, gamma=0.4)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    fed_w = donor_weights(1, [None] * 4, p, alpha=0.0, beta=1.0)
    fed = aggregate(thetas, 1, fed_w, gamma=float(p[1]))
    return {"thetas": [{k: v.tolist() for k, v in t.items()} for t in thetas],
            "w": w.tolist(), "gamma": 0.4,
            "mixed": {k: v.tolist() for k, v in out.items()},
            "fedavg_p": p.tolist(), "fedavg_w": fed_w.tolist(),
            "fedavg_mixed": {k: v.tolist() for k, v in fed.items()}}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = synthetic_df()
    roles = build_roles(df)
    cfg = base_cfg()
    ref = {"seed": SEED, "roles": jsonable(roles),
           "df_sha": hashlib.sha256(pd.util.hash_pandas_object(df, index=False)
                                    .to_numpy().tobytes()).hexdigest()[:16],
           "aggregation": capture_aggregation(), "builders": {}, "end_to_end": {}}

    builders = {
        "e1": (lambda c: build_clients(df, roles, c, SEED), cfg, ARMS),
        "e1m_b": (lambda c: build_matched_clients(df, roles, c, SEED, "b")[:2], cfg, ARMS),
    }
    cfg_e5 = copy.deepcopy(cfg)
    cfg_e5["e5"] = {"P": 1, "M": 1, "condition": "P1M1", "s_config": "primary", "s_exclude": ()}
    cfg_e5["arms"] = E5_ARMS
    builders["e5_P1M1"] = (lambda c: build_e5_clients(df, roles, c, SEED)[:2], cfg_e5, E5_ARMS)

    for name, (build, c, arms) in builders.items():
        clients, groups = build(c)
        ref["builders"][name] = capture_builder(name, clients, groups, roles, c, arms)
        print(f"  captured builder {name}: {len(clients)} clients, "
              f"{[len(x['train']) for x in clients]} training rows")

    # small end-to-end run through the current orchestrator
    for name, c, builder in [("e1", cfg, None),
                             ("e5_P1M1", cfg_e5, lambda s: build_e5_clients(df, roles, cfg_e5, s)[:2])]:
        out = run_seed(df, roles, copy.deepcopy(c), SEED, folds=("val", "test"),
                       headroom=False, builder=builder)
        m = pd.DataFrame(out["metrics"]).sort_values(
            ["receiver", "arm", "fold", "budget", "metric"]).reset_index(drop=True)
        ref["end_to_end"][name] = {
            "n_rows": len(m),
            "metrics": [{"receiver": r.receiver, "arm": r.arm, "fold": r.fold, "budget": int(r.budget),
                         "metric": r.metric, "value": float(r.value)} for r in m.itertuples()],
            "weights": sorted([{"receiver": w["receiver"], "arm": w["arm"], "donor": w["donor"],
                                "weight": float(w["weight"]), "gamma": float(w["gamma"]),
                                "fallback": w["fallback"]} for w in out["weights"]],
                              key=lambda x: (x["receiver"], x["arm"], x["donor"])),
        }
        print(f"  captured end-to-end {name}: {len(m)} metric rows")

    path = os.path.join(OUT, "reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(ref), f, indent=1, sort_keys=True)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
