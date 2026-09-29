"""E5 analysis (docs/PROTOCOL_E5.md §7-§10), from saved artefacts. Test fold.

Primary comparisons, by dataset x condition x S-configuration, and how they
change between P0/P1 and M0/M1. Delta% = 100*(RMSE_a - RMSE_b)/RMSE_b in
original units, macro-averaged over receivers within a seed; mean and 95%
t-interval over seeds; delta = 2%.

Run:  python -m analysis.e5_analysis --stage eval > results/e5/analysis.txt
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from stats import DELTA_PCT, decide, seed_interval

PRIMARY = [("combined-Q", "coverage-W_H", "does population similarity add value?"),
           ("combined-Q", "population-S", "does coverage add value?"),
           ("combined-Q", "combined-Q-marginal", "does PAIRWISE add beyond marginal+S?"),
           ("population-S", "uniform-donor", "does the population signal help?"),
           ("combined-Q", "uniform-donor", "does the combined method help overall?")]
CONDS = ["P0M0", "P0M1", "P1M0", "P1M1"]
DATASETS = ["concrete", "wine", "kin8nm", "protein"]


def load(stage):
    runs = {}
    for r in sorted(glob.glob(f"results/e5_{stage}_*/*/")):
        cfg = json.load(open(os.path.join(r, "config.json")))
        e5 = cfg["e5"]
        tabs = {k: pd.read_csv(os.path.join(r, f"{k}.csv"))
                for k in ["metrics", "scores", "candidates"]
                if os.path.exists(os.path.join(r, f"{k}.csv"))}
        runs[(cfg["dataset"], e5["condition"], e5["s_config"])] = (cfg, tabs)
    return runs


def piv(m, tp, fold="test"):
    r = m[(m.metric == "rmse") & (m.fold == fold) & (m.timepoint == tp)]
    return r.pivot_table(index=["seed", "receiver"], columns="arm", values="value")


def ci(t, a, b):
    return seed_interval(100 * (t[a] - t[b]) / t[b])


def fmt(c):
    return f"{c['mean_pct']:+6.2f} [{c['lo']:+6.2f},{c['hi']:+6.2f}] {decide(c['lo'], c['hi'])[:12]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="eval")
    args = ap.parse_args()
    runs = load(args.stage)
    if not runs:
        print(f"no runs found for stage {args.stage}")
        return
    datasets = [d for d in DATASETS if any(k[0] == d for k in runs)]
    sconfs = sorted({k[2] for k in runs})
    print(f"E5 analysis - stage {args.stage}. Test fold; Delta% of the comparator's RMSE; "
          f"seed = replication unit; delta = {DELTA_PCT}%.")
    print(f"Datasets {datasets}; conditions {CONDS}; S configurations {sconfs}.\n")

    print("=" * 110)
    print("PRIMARY COMPARISONS (timepoint t2)")
    for a, b, q in PRIMARY:
        print(f"\n  {a} vs {b}   -- {q}")
        for sconf in sconfs:
            print(f"    S configuration: {sconf}")
            for name in datasets:
                cells = []
                for cond in CONDS:
                    key = (name, cond, sconf)
                    if key not in runs:
                        continue
                    t = piv(runs[key][1]["metrics"], "t2")
                    cells.append(f"{cond} {fmt(ci(t, a, b))}")
                if cells:
                    print(f"      {name:9s} " + " | ".join(cells))

    print("\n" + "=" * 110)
    print("SAME COMPARISONS AT t1 (immediately after mixing)")
    for a, b, _ in PRIMARY:
        print(f"\n  {a} vs {b}")
        for sconf in sconfs:
            for name in datasets:
                cells = []
                for cond in CONDS:
                    key = (name, cond, sconf)
                    if key in runs:
                        t = piv(runs[key][1]["metrics"], "t1")
                        cells.append(f"{cond} {fmt(ci(t, a, b))}")
                if cells:
                    print(f"      {sconf:7s} {name:9s} " + " | ".join(cells))

    print("\n" + "=" * 110)
    print("MACRO RMSE (original units, mean over seeds x receivers, t2) and harmful transfer")
    for sconf in sconfs:
        for name in datasets:
            for cond in CONDS:
                key = (name, cond, sconf)
                if key not in runs:
                    continue
                t = piv(runs[key][1]["metrics"], "t2")
                arms = [c for c in t.columns]
                means = t.mean()
                harmed = {a: int((t[a] > t["local-only"]).sum()) for a in arms if a != "local-only"}
                per_rec = t.groupby(level="receiver").mean()
                worst = {a: float((per_rec[a] - per_rec["local-only"]).max()) for a in harmed}
                print(f"  {sconf:7s} {name:9s} {cond}: " + " ".join(
                    f"{a.replace('combined-','Q-').replace('coverage-','cov-')}={means[a]:.3f}" for a in arms))
                print(f"          receiver-seeds harmed vs local-only (of {len(t)}): " + " ".join(
                    f"{a.replace('combined-','Q-').replace('coverage-','cov-')}={harmed[a]}" for a in harmed)
                      + f"; worst per-receiver loss {max(worst.values()):+.3f}")

    print("\n" + "=" * 110)
    print("SCORE VARIATION AND REALISED WEIGHT VARIATION (mean over receiver-seeds)")
    print("  score range = max-min across the 3 donors; weight range likewise (uniform = 0.333)")
    for sconf in sconfs:
        for name in datasets:
            for cond in CONDS:
                key = (name, cond, sconf)
                if key not in runs:
                    continue
                sc = runs[key][1]["scores"]
                p = sc.pivot_table(index=["seed", "receiver", "donor"], columns="score", values="value")
                parts = []
                for s in ["W_marg", "W_H", "S", "Q", "Q_marg"]:
                    g = p[s].groupby(level=["seed", "receiver"])
                    rng = (g.max() - g.min()).mean()
                    w = p[s] / g.transform("sum")
                    wr = (w.groupby(level=["seed", "receiver"]).max()
                          - w.groupby(level=["seed", "receiver"]).min()).mean()
                    parts.append(f"{s} {rng:.3f}/{wr:.3f}")
                print(f"  {sconf:7s} {name:9s} {cond}: " + "  ".join(parts))

    print("\n" + "=" * 110)
    print("DONOR-SCORE AGREEMENT WITH MEASURED TRANSFER BENEFIT U (retrospective; never used to build scores)")
    print("  top donor by score == top donor by U, of receiver-seeds; and mean Kendall tau")
    for sconf in sconfs:
        for name in datasets:
            for cond in CONDS:
                key = (name, cond, sconf)
                if key not in runs:
                    continue
                sc = runs[key][1]["scores"]
                p = sc.pivot_table(index=["seed", "receiver", "donor"],
                                   columns="score", values="value")
                u = sc[sc.score == "W_H"].set_index(["seed", "receiver", "donor"])["U_t2"]
                parts = []
                for s in ["W_marg", "W_H", "S", "Q", "Q_marg"]:
                    hit, n, taus = 0, 0, []
                    for idx, g in p[s].groupby(level=["seed", "receiver"]):
                        donors = [k[2] for k in g.index]
                        uu = u.loc[idx].reindex(donors)          # align by donor, not by order
                        if g.isna().any() or uu.isna().any():
                            continue
                        n += 1
                        hit += int(donors[int(np.argmax(g.values))] == donors[int(np.argmax(uu.values))])
                        taus.append(kendalltau(g.values, uu.values).statistic)
                    parts.append(f"{s} {hit}/{n} tau {np.nanmean(taus):+.2f}")
                print(f"  {sconf:7s} {name:9s} {cond}: " + "  ".join(parts))

    print("\n" + "=" * 110)
    print("VALIDATION-SELECTED DONOR (diagnostic): best single-donor mix chosen on val, scored on test")
    for sconf in sconfs:
        for name in datasets:
            for cond in CONDS:
                key = (name, cond, sconf)
                if key not in runs or "candidates" not in runs[key][1]:
                    continue
                cand, m = runs[key][1]["candidates"], runs[key][1]["metrics"]
                cv = cand[(cand.metric == "rmse") & (cand.fold == "val") & (cand.timepoint == "t2")]
                ct = cand[(cand.metric == "rmse") & (cand.fold == "test") & (cand.timepoint == "t2")]
                pv = cv.pivot_table(index=["seed", "receiver"], columns="donor", values="value")
                pt = ct.pivot_table(index=["seed", "receiver"], columns="donor", values="value")
                t = piv(m, "t2")
                pick = pv.idxmin(axis=1)
                sel = pd.Series([pt.loc[i, d] for i, d in pick.items()], index=pick.index)
                c1 = seed_interval(100 * (sel - t["local-only"]) / t["local-only"])
                c2 = seed_interval(100 * (sel - t["uniform-donor"]) / t["uniform-donor"])
                print(f"  {sconf:7s} {name:9s} {cond}: vs local-only {fmt(c1)}; "
                      f"vs uniform-donor {fmt(c2)}")


if __name__ == "__main__":
    main()
