"""E1M analysis (CLAUDE.md §10, E1M), from saved artefacts, test fold.

Order: (1) U same-partition vs other-partition, within receiver and seed, then
by seed; (2) every arm vs uniform-donor, §8.1 intervals and delta decision,
with the correlation-matching |diff| as a covariate column; (3) top-ranked
donor vs a uniformly drawn donor; (4) oracle vs validation gap; (5) parameter
geometry for hypotheses A / B. The seed is the replication unit (n = 10).

Run:  python -m analysis.e1m_analysis > results/e1m/analysis.txt
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from loop import ARMS
from stats import DELTA_PCT, seed_interval

ORDER = ["kin8nm", "concrete", "protein", "wine"]          # ascending correlation-matching |diff|


def verdict(ci, delta=DELTA_PCT):
    if -delta < ci["lo"] and ci["hi"] < delta:
        return "negligible"
    if ci["hi"] < -delta or ci["lo"] > delta:
        return "meaningful"
    return "inconclusive"


def fmt(ci):
    return f"{ci['mean_pct']:+7.3f} ± {ci['hi'] - ci['mean_pct']:6.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"


def load():
    runs = {}
    for r in glob.glob("results/e1m/*/"):
        cfg = json.load(open(os.path.join(r, "config.json")))
        t = {k: pd.read_csv(os.path.join(r, f"{k}.csv")) for k in ["metrics", "candidates", "scores", "geometry"]}
        runs[(cfg["dataset"], cfg["e1m"]["condition"])] = (cfg, t)
    return runs


def piv(df, key, tp, fold="test"):
    r = df[(df.metric == "rmse") & (df.fold == fold) & (df.timepoint == tp)]
    return r.pivot_table(index=["seed", "receiver"], columns=key, values="value")


def q1(t, tp):
    """U(same) - mean U(other), in percent of the receiver's local-only RMSE."""
    s = t["scores"][t["scores"].score == "W_H"]
    local = piv(t["metrics"], "arm", tp)["local-only"]
    rows, ties, same_best = [], 0, 0
    for (seed, rec), g in s.groupby(["seed", "receiver"]):
        u = g[f"U_{tp}"]
        same, other = u[g.same_group.values], u[~g.same_group.values]
        diff = same.mean() - other.mean()
        ties += int(np.isclose(diff, 0.0))
        same_best += int(same.max() > other.max())
        rows.append({"seed": seed, "receiver": rec, "d": 100 * diff / local.loc[(seed, rec)],
                     "n_same": len(same), "n_other": len(other)})
    d = pd.DataFrame(rows).set_index(["seed", "receiver"])
    return seed_interval(d["d"]), len(d), int(d.n_same.iloc[0]), int(d.n_other.iloc[0]), ties, same_best


def main():
    runs = load()
    conds = sorted({c for _, c in runs})
    print("E1M analysis - test fold; Δ% = 100·(RMSE_a − RMSE_b)/RMSE_b; seed = replication unit (n=10); "
          f"δ = {DELTA_PCT}%. Datasets ordered by correlation-matching |diff| (covariate).")

    print("\n(1) U same-partition − mean U other-partition, % of local-only RMSE (positive = same-partition "
          "donor transfers better)")
    for cond in conds:
        for tp in ["t1", "t2"]:
            for name in ORDER:
                if (name, cond) not in runs:
                    continue
                cfg, t = runs[(name, cond)]
                ci, n, ns, no, ties, best = q1(t, tp)
                print(f"  ({cond}) {tp} {name:9s} |diff| {cfg['e1m']['corr_diff']:.3f}  {fmt(ci)}  -> "
                      f"{verdict(ci)};  receiver-seeds {n} (donors/receiver: {ns} same, {no} other); "
                      f"ties {ties}; same-partition donor is the single best donor in {best}/{n}")

    print("\n(2) every arm vs uniform-donor (Δ% of uniform-donor RMSE)")
    for cond in conds:
        for tp in ["t1", "t2"]:
            print(f"  condition ({cond}) {tp}")
            print(f"    {'arm':23s} " + "  ".join(f"{n:>34s}" for n in ORDER if (n, cond) in runs))
            print(f"    {'|diff| covariate':23s} " + "  ".join(
                f"{runs[(n, cond)][0]['e1m']['corr_diff']:>34.3f}" for n in ORDER if (n, cond) in runs))
            for arm in [a for a in ARMS if a != "uniform-donor"]:
                cells = []
                for name in ORDER:
                    if (name, cond) not in runs:
                        continue
                    a = piv(runs[(name, cond)][1]["metrics"], "arm", tp)
                    ci = seed_interval(100 * (a[arm] - a["uniform-donor"]) / a["uniform-donor"])
                    cells.append(f"{ci['mean_pct']:+6.2f} [{ci['lo']:+6.2f},{ci['hi']:+6.2f}] {verdict(ci)[:5]:>5s}")
                print(f"    {arm:23s} " + "  ".join(f"{c:>34s}" for c in cells))

    print("\n(3) top-ranked donor (single-donor mix) vs a uniformly drawn donor, and (4) oracle vs validation")
    for cond in conds:
        for tp in ["t1", "t2"]:
            for name in ORDER:
                if (name, cond) not in runs:
                    continue
                t = runs[(name, cond)][1]
                cand = piv(t["candidates"], "donor", tp)
                arms = piv(t["metrics"], "arm", tp)
                sc = t["scores"].pivot_table(index=["seed", "receiver", "donor"], columns="score", values="value")
                drawn = cand.mean(axis=1)
                parts = []
                for score in ["s", "W_H"]:
                    top = sc[score].groupby(level=["seed", "receiver"]).idxmax().map(lambda k: k[2])
                    top_rmse = pd.Series([cand.loc[i, dn] for i, dn in top.items()], index=top.index)
                    ci = seed_interval(100 * (top_rmse - drawn) / drawn)
                    parts.append(f"top-{score} vs drawn {ci['mean_pct']:+.2f} [{ci['lo']:+.2f},{ci['hi']:+.2f}] {verdict(ci)}")
                best = cand.min(axis=1)
                vpick = piv(t["candidates"], "donor", tp, "val").idxmin(axis=1)
                vsel = pd.Series([cand.loc[i, dn] for i, dn in vpick.items()], index=vpick.index)
                ci_o = seed_interval(100 * (best - arms["uniform-donor"]) / arms["uniform-donor"])
                ci_v = seed_interval(100 * (vsel - arms["uniform-donor"]) / arms["uniform-donor"])
                print(f"  ({cond}) {tp} {name:9s} " + "; ".join(parts))
                print(f"  {'':17s} oracle best donor vs uniform {fmt(ci_o)} {verdict(ci_o)}; "
                      f"val-picked donor vs uniform {fmt(ci_v)} {verdict(ci_v)}")

    print("\n(5) parameter geometry (L2, flattened parameters; mixtures before adaptation)")
    for cond in conds:
        for name in ORDER:
            if (name, cond) not in runs:
                continue
            g = runs[(name, cond)][1]["geometry"]
            cc = g[g.kind == "client-client"].dist
            c0 = g[g.kind == "client-theta0"].dist
            arms = g[g.kind == "arm"].groupby("a")[["to_centroid", "to_uniform_mix", "to_receiver_local"]].mean()
            singles = g[g.kind == "single"][["to_centroid", "to_receiver_local"]].mean()
            print(f"  ({cond}) {name:9s} client-client {cc.mean():.3f} (min {cc.min():.3f}, max {cc.max():.3f}); "
                  f"client-theta0 {c0.mean():.3f}; ratio {cc.mean() / c0.mean():.2f}")
            print(f"      single-donor mix: to centroid {singles.to_centroid:.3f}, to receiver local "
                  f"{singles.to_receiver_local:.3f}")
            print("      arms (to centroid / to uniform mix / to receiver local): " + "; ".join(
                f"{a} {r.to_centroid:.3f}/{r.to_uniform_mix:.3f}/{r.to_receiver_local:.3f}"
                for a, r in arms.iterrows()))


if __name__ == "__main__":
    main()
