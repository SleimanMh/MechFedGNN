"""E5 Stage A - synthetic sanity check (docs/PROTOCOL_E5.md §9).

Checks the pipeline and that the intended population differences exist.
Success for S or Q is NOT required and is not used to choose anything.

Synthetic problem:
  - ONE shared outcome function across clients (so any client difference comes
    from the covariate distribution, not from a different y|x rule);
  - overlapping but different covariate distributions (the P1 mechanism);
  - missingness independent of feature values (M0 / M1 as in the protocol);
  - predictive relevance that VARIES ACROSS REGIONS of the covariate space:
    y = f0 + 2*sin(2*f1)*1[f0 <= 0] + 0.5*f2*1[f0 > 0] + ... , so which
    feature matters depends on where a client sits in f0.

Run:  python -m analysis.e5_stage_a > results/e5/stage_a.txt
"""
import copy

import numpy as np
import pandas as pd
import yaml

from data.e5 import FOLDS, build_e5_clients, partition_column
from data.inject import build_roles
from loop import E5_ARMS, run_seed, summaries
from signatures import population_similarity
from stats import DELTA_PCT, decide, seed_interval

N, D = 4000, 8


def synthetic(seed=0):
    """Shared outcome function; relevance varies with the region of f0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]          # correlated pair -> panel structure
    X[:, 3] = 0.8 * X[:, 2] + 0.6 * X[:, 3]
    left = X[:, 0] <= 0
    y = (X[:, 0]
         + np.where(left, 2.0 * np.sin(2 * X[:, 1]), 0.0)
         + np.where(left, 0.0, 0.5 * X[:, 2] ** 2)
         + 0.3 * X[:, 4] + 0.2 * rng.standard_normal(N))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(D)])
    df["target"] = y
    return df


def cfg_for(base, P, M, sconf, pcol, seeds):
    cfg = copy.deepcopy(base)
    cfg["corrections"] = {"dup_groups": True, "shared_scaler": True}
    cfg["e5"] = {"P": P, "M": M, "condition": f"P{P}M{M}", "s_config": sconf,
                 "s_exclude": (pcol,) if sconf == "robust" else (),
                 "partition_characteristic": f"f{pcol}", "frac_low": base["clients"]["frac_low"],
                 "stage": "A"}
    cfg["arms"] = E5_ARMS
    cfg["seeds"] = seeds
    cfg["model"] = {**base["model"], "local_steps": 200, "adapt_budget": 10}
    return cfg


def main():
    base = yaml.safe_load(open("configs/defaults.yaml"))
    seeds = base["e5_plan"]["dev_seeds"]
    df = synthetic()
    roles = build_roles(df)
    pcol = partition_column(df, roles)
    feats = roles["features"]
    print("E5 Stage A - synthetic sanity check. Success for S or Q is NOT required.\n")
    print(f"  n={N} d={D}; one shared outcome function; relevance switches at f0=0")
    print(f"  maskable {[feats[i] for i in roles['maskable']]}; always_observed "
          f"{[feats[i] for i in roles['always_observed']]}")
    print(f"  partition characteristic (declared rule): {feats[pcol]}")

    print("\n1) Do the intended population differences exist?")
    for P in (0, 1):
        cfg = cfg_for(base, P, 1, "primary", pcol, seeds)
        clients, _, _ = build_e5_clients(df, roles, cfg, seeds[0])
        med = np.median(np.concatenate([c["X"][:, pcol] for c in clients]))
        share = [float(np.mean(c["X"][c["train"]][:, pcol] <= med)) for c in clients]
        summ = [summaries(c, roles) for c in clients]
        S = [population_similarity(summ[i]["hist"], summ[j]["hist"])
             for i in range(len(clients)) for j in range(i + 1, len(clients))]
        ymean = [float(c["y"][c["train"]].mean()) for c in clients]
        print(f"   P{P}: below-median share per client {np.round(share, 2).tolist()}; "
              f"S off-diagonal {min(S):.3f}-{max(S):.3f} (spread {max(S) - min(S):.3f}); "
              f"train y mean {np.round(ymean, 2).tolist()}")
    print("   (P1 shares should track frac_low and keep overlap; S spread should widen.)")

    print("\n2) Pipeline runs and M0/M1 differ only in joint structure")
    res = {}
    for P in (0, 1):
        for M in (0, 1):
            cfg = cfg_for(base, P, M, "primary", pcol, seeds)
            builder = lambda seed, c=cfg: build_e5_clients(df, roles, c, seed)[:2]
            outs = [run_seed(df, roles, cfg, s, folds=("val", "test"), headroom=False,
                             builder=builder) for s in seeds]
            m = pd.DataFrame([r for o in outs for r in o["metrics"]])
            sc = pd.DataFrame([r for o in outs for r in o["scores"]])
            res[(P, M)] = (m, sc)
            piv = sc.pivot_table(index=["seed", "receiver", "donor"], columns="score", values="value")
            print(f"   P{P}M{M}: W_marg {piv['W_marg'].mean():.4f}  W_H {piv['W_H'].mean():.4f}  "
                  f"S {piv['S'].mean():.4f}  Q {piv['Q'].mean():.4f}  Q_marg {piv['Q_marg'].mean():.4f}")
    for P in (0, 1):
        a = res[(P, 0)][1].pivot_table(index=["seed", "receiver", "donor"], columns="score", values="value")
        b = res[(P, 1)][1].pivot_table(index=["seed", "receiver", "donor"], columns="score", values="value")
        same = {k: bool(np.allclose(a[k], b[k])) for k in ["W_marg", "S", "Q_marg"]}
        diff = not np.allclose(a["W_H"], b["W_H"])
        print(f"   P{P}: M0 vs M1 identical for {same}; W_H differs: {diff}  "
              "(designed: only joint structure changes)")

    print("\n3) Arm contrasts (test fold, t2) - reported, not required to favour any arm")
    for (P, M), (m, _) in res.items():
        r = m[(m.metric == "rmse") & (m.fold == "test") & (m.timepoint == "t2")]
        t = r.pivot_table(index=["seed", "receiver"], columns="arm", values="value")
        parts = []
        for a, b in [("population-S", "uniform-donor"), ("combined-Q", "combined-Q-marginal"),
                     ("combined-Q", "uniform-donor")]:
            ci = seed_interval(100 * (t[a] - t[b]) / t[b])
            parts.append(f"{a} vs {b} {ci['mean_pct']:+.2f} [{ci['lo']:+.2f},{ci['hi']:+.2f}] "
                         f"{decide(ci['lo'], ci['hi'], DELTA_PCT)}")
        print(f"   P{P}M{M}: " + "; ".join(parts))
    print("\nStage A is a pipeline and construction check only. No budget, arm or setting is "
          "chosen from these numbers.")


if __name__ == "__main__":
    main()
