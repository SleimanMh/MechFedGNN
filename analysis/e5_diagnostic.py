"""E5 selection diagnostic - EXPLORATORY, post hoc, saved artefacts only.

No retraining, no injector change, no tuning. Answers:
  Is there a useful donor-selection opportunity beyond uniform aggregation,
  and can the available validation data reliably identify it?

Unit of replication: the SEED. Receivers are averaged within a seed before any
interval is formed; receivers, donor pairs and conditions are not treated as
independent replications.

The test-selected donor is an optimistic retrospective reference for the
available single-donor candidates - not a deployable method, and not a ceiling
for aggregation methods in general.

Run:  python -m analysis.e5_diagnostic > results/e5/diagnostic.txt
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from stats import DELTA_PCT, decide, seed_interval

DATASETS = ["concrete", "wine", "kin8nm", "protein"]
CONDS = ["P0M0", "P0M1", "P1M0", "P1M1"]


def load(sconf="primary"):
    runs = {}
    for r in sorted(glob.glob("results/e5_eval_*/*/")):
        cfg = json.load(open(os.path.join(r, "config.json")))
        if cfg["e5"]["s_config"] != sconf:
            continue
        runs[(cfg["dataset"], cfg["e5"]["condition"])] = (
            pd.read_csv(os.path.join(r, "metrics.csv")),
            pd.read_csv(os.path.join(r, "candidates.csv")))
    return runs


def tables(m, c, tp):
    """(arms, cand) pivots for one timepoint: index (seed, receiver)."""
    a = m[(m.metric == "rmse") & (m.timepoint == tp)]
    cc = c[(c.metric == "rmse") & (c.timepoint == tp)]
    out = {}
    for fold in ["val", "test"]:
        out[("arm", fold)] = a[a.fold == fold].pivot_table(
            index=["seed", "receiver"], columns="arm", values="value")
        out[("cand", fold)] = cc[cc.fold == fold].pivot_table(
            index=["seed", "receiver"], columns="donor", values="value")
    return out


def selectors(t):
    """The six quantities, all on the TEST fold, index (seed, receiver)."""
    at, av = t[("arm", "test")], t[("arm", "val")]
    ct, cv = t[("cand", "test")], t[("cand", "val")]
    donors = list(ct.columns)
    pick_d = cv.idxmin(axis=1)
    sel_d = pd.Series([ct.loc[i, d] for i, d in pick_d.items()], index=pick_d.index)
    # donors + local-only, chosen on validation
    cvl = cv.copy()
    cvl["local-only"] = av["local-only"]
    ctl = ct.copy()
    ctl["local-only"] = at["local-only"]
    pick_dl = cvl.idxmin(axis=1)
    sel_dl = pd.Series([ctl.loc[i, d] for i, d in pick_dl.items()], index=pick_dl.index)
    return {"local-only": at["local-only"], "uniform-donor": at["uniform-donor"],
            "random donor (expected)": ct[donors].mean(axis=1),
            "test-best donor (retrospective)": ct[donors].min(axis=1),
            "val-selected donor": sel_d,
            "val-selected donor or local": sel_dl}, pick_d, pick_dl, ct, cv, at, av


def ci(a, b):
    return seed_interval(100 * (a - b) / b)


def fmt(c):
    return f"{c['mean_pct']:+6.2f} [{c['lo']:+6.2f},{c['hi']:+6.2f}] {decide(c['lo'], c['hi'])}"


def main():
    print("E5 SELECTION DIAGNOSTIC - EXPLORATORY, post hoc, saved artefacts only.")
    print("No retraining, no injector change, no tuning. Seed = replication unit (n=10);")
    print(f"receivers averaged within seed first. Practical margin delta = {DELTA_PCT}%.")
    print("The test-best donor is an OPTIMISTIC RETROSPECTIVE reference for the available")
    print("single-donor candidates - not a method, and not a ceiling for aggregation.\n")

    for sconf in ["primary", "robust"]:
        runs = load(sconf)
        if not runs:
            continue
        print("=" * 118)
        print(f"S CONFIGURATION: {sconf}")
        for tp in ["t2", "t1"]:
            print(f"\n--- (1) COLLABORATION vs SELECTION, timepoint {tp} "
                  f"({'post-adaptation' if tp == 't2' else 'immediate'}); "
                  "Delta% vs UNIFORM-DONOR, matched self-weight and budget")
            print(f"    {'dataset':9s} {'cond':5s} " + "".join(
                f"{k:>34s}" for k in ["local-only", "random donor", "test-best (retro)",
                                      "val-selected donor", "val-sel donor|local"]))
            for name in DATASETS:
                for cond in CONDS:
                    if (name, cond) not in runs:
                        continue
                    t = tables(*runs[(name, cond)], tp)
                    q, *_ = selectors(t)
                    cells = [fmt(ci(q[k], q["uniform-donor"])) for k in
                             ["local-only", "random donor (expected)",
                              "test-best donor (retrospective)", "val-selected donor",
                              "val-selected donor or local"]]
                    print(f"    {name:9s} {cond:5s} " + "".join(f"{c:>34s}" for c in cells))

        print(f"\n--- (2) STABILITY OF DONOR PREFERENCES (t2; validation vs test)")
        print(f"    {'dataset':9s} {'cond':5s} {'tau(val,test)':>22s} {'val-best=test-best':>22s} "
              f"{'val gap 1st-2nd %':>20s} {'improve-vs-local disagree':>28s} {'ties':>6s}")
        for name in DATASETS:
            for cond in CONDS:
                if (name, cond) not in runs:
                    continue
                t = tables(*runs[(name, cond)], "t2")
                _, pick_d, _, ct, cv, at, av = selectors(t)
                donors = list(ct.columns)
                per_seed = {}
                ties = 0
                for (seed, rec) in ct.index:
                    # a receiver has no self-candidate: drop its own (NaN) column
                    v = cv.loc[(seed, rec), donors].dropna()
                    s = ct.loc[(seed, rec), donors].reindex(v.index)
                    ties += int(v.duplicated().any() or s.duplicated().any())
                    tau = kendalltau(v.values, s.values).statistic
                    order = np.argsort(v.values)
                    gap = 100 * (v.values[order[1]] - v.values[order[0]]) / v.values[order[0]]
                    agree_best = int(v.idxmin() == s.idxmin())
                    dis = np.mean((v.values < av.loc[(seed, rec), "local-only"]) !=
                                  (s.values < at.loc[(seed, rec), "local-only"]))
                    per_seed.setdefault(seed, []).append((tau, agree_best, gap, dis))
                arr = {k: np.array([np.mean([r[i] for r in v]) for v in per_seed.values()])
                       for i, k in enumerate(["tau", "best", "gap", "dis"])}
                f = lambda a: f"{a.mean():+.2f} +-{1.96 * a.std(ddof=1) / np.sqrt(len(a)):.2f}"
                print(f"    {name:9s} {cond:5s} {f(arr['tau']):>22s} "
                      f"{arr['best'].mean():>21.0%} {arr['gap'].mean():>19.2f}% "
                      f"{arr['dis'].mean():>27.0%} {ties:>6d}")

        print(f"\n--- (2b) DONOR RANKING BEFORE vs AFTER ADAPTATION (test fold, t1 vs t2)")
        for name in DATASETS:
            row = []
            for cond in CONDS:
                if (name, cond) not in runs:
                    continue
                m, c = runs[(name, cond)]
                a = tables(m, c, "t1")[("cand", "test")]
                b = tables(m, c, "t2")[("cand", "test")]
                donors = list(a.columns)
                per_seed = {}
                for idx in a.index:
                    x = a.loc[idx, donors].dropna()
                    y = b.loc[idx, donors].reindex(x.index)
                    tau = kendalltau(x.values, y.values).statistic
                    same = int(x.idxmin() == y.idxmin())
                    per_seed.setdefault(idx[0], []).append((tau, same))
                tau_m = np.mean([np.mean([r[0] for r in v]) for v in per_seed.values()])
                same_m = np.mean([np.mean([r[1] for r in v]) for v in per_seed.values()])
                row.append(f"{cond} tau {tau_m:+.2f} best-same {same_m:.0%}")
            print(f"    {name:9s} " + " | ".join(row))
    print("\nLIMITATION: per-example predictions were not saved for E5 (function.csv holds RMS")
    print("summaries only), so the optional bootstrap over validation examples cannot be run")
    print("from existing artefacts. It would require re-evaluating saved models - not done here.")


if __name__ == "__main__":
    main()
