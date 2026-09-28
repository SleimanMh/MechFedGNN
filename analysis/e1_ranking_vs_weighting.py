"""EXPLORATORY post-hoc analysis of E1 (not pre-registered). Saved artefacts only.

Reads results/e1/<run>/{metrics,candidates,scores}.csv and config.json.
Everything here is computed on the TEST fold unless stated; mixing ratio
(gamma = 0.5 for uniform-donor, every score arm and every single-donor
candidate) and adaptation budget (adapt_budget) are the ones E1 used, so they
are matched throughout.

Effect unit (CLAUDE.md §8.1): delta_pct = 100 * (RMSE_a - RMSE_b) / RMSE_b per
receiver-seed; averaged over receivers within each seed; mean and 95%
t-interval over the n = 10 seeds (the seed is the replication unit, not the
40 receiver-seeds).

Donor weights are not saved; with alpha = beta = 1 and no fallback (E1 report:
the fallback never fired) they are exactly w_j = q_j / sum_l q_l, so they are
recomputed from scores.csv.

Run:  python -m analysis.e1_ranking_vs_weighting > results/e1/exploratory_ranking_vs_weighting.txt
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from stats import DELTA_PCT, seed_interval

SCORE_OF_ARM = {"marginal-rate": "rate", "missingness-similarity": "s", "coverage-W_H": "W_H",
                "coverage-W_C": "W_C", "population-S": "S", "combined-Q": "Q"}


def verdict(ci, delta=DELTA_PCT):
    """'negligible' only if the whole interval lies inside (-delta, +delta)."""
    if -delta < ci["lo"] and ci["hi"] < delta:
        return "negligible"
    if ci["hi"] < -delta or ci["lo"] > delta:
        return "meaningful"
    return "inconclusive"


def fmt(ci):
    hw = ci["hi"] - ci["mean_pct"]
    return f"{ci['mean_pct']:+7.3f}% ± {hw:6.3f}  [{ci['lo']:+.3f}, {ci['hi']:+.3f}]  n={ci['seeds']}"


def rel(a, b):
    return 100.0 * (a - b) / b


def load(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    m = pd.read_csv(os.path.join(run, "metrics.csv"))
    c = pd.read_csv(os.path.join(run, "candidates.csv"))
    s = pd.read_csv(os.path.join(run, "scores.csv"))
    return cfg, m, c, s


def arm_rmse(m, tp):
    r = m[(m.metric == "rmse") & (m.fold == "test") & (m.timepoint == tp)]
    return r.pivot_table(index=["seed", "receiver"], columns="arm", values="value")


def cand_rmse(c, tp, fold="test"):
    r = c[(c.metric == "rmse") & (c.fold == fold) & (c.timepoint == tp)]
    return r.pivot_table(index=["seed", "receiver"], columns="donor", values="value")


def item2(m):
    print("ITEM 2 - score arm vs uniform-donor, test fold, RMSE")
    for tp in ["t1", "t2"]:
        a = arm_rmse(m, tp)
        for arm in SCORE_OF_ARM:
            ci = seed_interval(rel(a[arm], a["uniform-donor"]))
            print(f"  {tp} {arm:23s} {fmt(ci)}  -> {verdict(ci)}")


def item3(m, c, s):
    print("ITEM 3 - ranking vs weighting (EXPLORATORY)")
    piv = s.pivot_table(index=["seed", "receiver", "donor"], columns="score", values="value")
    print("  (a) score range and weight range across the 3 donors, per receiver-seed "
          "(median [max]); uniform weight = 0.333")
    for arm, sc in SCORE_OF_ARM.items():
        g = piv[sc].groupby(level=["seed", "receiver"])
        s_rng = g.max() - g.min()
        rel_rng = s_rng / g.mean()
        w = piv[sc] / g.transform("sum")
        wg = w.groupby(level=["seed", "receiver"])
        w_rng = wg.max() - wg.min()
        print(f"      {arm:23s} score range {s_rng.median():.4f} [{s_rng.max():.4f}]  "
              f"relative {rel_rng.median():.3f}  weight range {w_rng.median():.4f} [{w_rng.max():.4f}]  "
              f"max weight {wg.max().median():.3f}")
    for tp in ["t1", "t2"]:
        cand = cand_rmse(c, tp)
        arms = arm_rmse(m, tp)
        uniform_pick = cand.mean(axis=1)                       # expected RMSE of a uniformly drawn donor
        best = cand.min(axis=1)
        print(f"  [{tp}] single-donor mixes (gamma 0.5, same budget):")
        for arm, sc in SCORE_OF_ARM.items():
            top = piv[sc].groupby(level=["seed", "receiver"]).idxmax().map(lambda t: t[2])
            top_rmse = pd.Series([cand.loc[i, d] for i, d in top.items()], index=top.index)
            ci_u = seed_interval(rel(top_rmse, uniform_pick))
            ci_l = seed_interval(rel(top_rmse, arms["local-only"]))
            pos = (top_rmse < arms["local-only"]).mean()
            is_best = np.isclose(top_rmse, best).mean()
            least_harm = (np.isclose(top_rmse, best) & (best >= arms["local-only"])).mean()
            print(f"      top by {sc:4s}: vs uniformly drawn donor {fmt(ci_u)} -> {verdict(ci_u)}")
            print(f"      {'':12s} vs local-only          {fmt(ci_l)} -> {verdict(ci_l)}; "
                  f"beats local in {pos:.0%} of receiver-seeds; is the best donor in {is_best:.0%}; "
                  f"is best-but-still-worse-than-local in {least_harm:.0%}")
        ci_ob = seed_interval(rel(arms["uniform-donor"], best))
        val_best = cand_rmse(c, tp, "val").idxmin(axis=1)
        val_sel = pd.Series([cand.loc[i, d] for i, d in val_best.items()], index=val_best.index)
        ci_vs = seed_interval(rel(arms["uniform-donor"], val_sel))
        print(f"      uniform mixture vs best candidate (oracle, test-picked)  {fmt(ci_ob)} -> {verdict(ci_ob)}")
        print(f"      uniform mixture vs val-selected single donor            {fmt(ci_vs)} -> {verdict(ci_vs)}")
        print(f"      any single donor beats local in {(best < arms['local-only']).mean():.0%} of "
              f"receiver-seeds (oracle)")


def main():
    for run in sorted(glob.glob("results/e1/*/")):
        cfg, m, c, s = load(run)
        print(f"\n######## {cfg['dataset']}  ({run})  gamma={cfg['aggregation']['gamma']}  "
              f"adapt_budget={cfg['model']['adapt_budget']}  seeds={len(cfg['seeds'])}")
        item2(m)
        item3(m, c, s)


if __name__ == "__main__":
    main()
