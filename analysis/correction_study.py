"""§16 correction study (EXPLORATORY): does each E1 / E1M conclusion survive?

Criteria declared in CLAUDE.md §16 before the run:
  (a) score arms vs uniform-donor survive if every contrast stays negligible
      (whole 95% interval inside ±2%);
  (b) mixing vs local-only survives if the t2 delta decision of fedavg,
      uniform-donor and each score arm is unchanged.
Then the functional comparison (predictions on receiver test inputs) and, as
a secondary observation, parameter distances.

Run:  python -m analysis.correction_study > results/correction_study.txt
"""
import glob
import json
import os

import pandas as pd

from loop import ARMS
from stats import DELTA_PCT, decide, seed_interval

SCORE_ARMS = ARMS[3:]
MIXING = ARMS[1:]
DATASETS = ["concrete", "wine", "kin8nm", "protein"]
VERSIONS = [("original", "e1", "e1m"), ("corrected-corr", "e1_corrected_corr", "e1m_corrected_corr"),
            ("corrected-random", "e1_corrected_random", "e1m_corrected_random")]


def load(exp, kind):
    out = {}
    for r in glob.glob(f"results/{exp}/*/"):
        cfg = json.load(open(os.path.join(r, "config.json")))
        key = cfg["dataset"] if kind == "e1" else (cfg["dataset"], cfg["e1m"]["condition"])
        tabs = {k: pd.read_csv(os.path.join(r, f"{k}.csv"))
                for k in ["metrics", "function", "geometry"] if os.path.exists(os.path.join(r, f"{k}.csv"))}
        out[key] = (cfg, tabs)
    return out


def rmse(m, tp):
    r = m[(m.metric == "rmse") & (m.fold == "test") & (m.timepoint == tp)]
    return r.pivot_table(index=["seed", "receiver"], columns="arm", values="value")


def ci(tab, a, b):
    return seed_interval(100 * (tab[a] - tab[b]) / tab[b])


def neg(c):
    return -DELTA_PCT < c["lo"] and c["hi"] < DELTA_PCT


def criterion_a(m):
    bounds, allneg = 0.0, True
    for tp in ["t1", "t2"]:
        t = rmse(m, tp)
        for arm in SCORE_ARMS:
            c = ci(t, arm, "uniform-donor")
            bounds = max(bounds, abs(c["lo"]), abs(c["hi"]))
            allneg &= neg(c)
    return allneg, bounds


def main():
    data = {v: (load(e1, "e1"), load(e1m, "e1m")) for v, e1, e1m in VERSIONS}
    print("§16 correction study - EXPLORATORY. Test fold; Δ% relative RMSE; seed = unit (n=10); "
          f"δ = {DELTA_PCT}%.\n")

    print("(a) score arms vs uniform-donor - survives if all contrasts negligible")
    for name in DATASETS:
        cells = []
        for v, _, _ in VERSIONS:
            e1, e1m = data[v]
            if name in e1:
                ok, b = criterion_a(e1[name][1]["metrics"])
                cells.append(f"{v}: E1 {'all negligible' if ok else 'NOT all negligible'} (max |bound| {b:.2f})")
            m_cells = [(c, criterion_a(e1m[(name, c)][1]["metrics"])) for c in "abc" if (name, c) in e1m]
            if m_cells:
                cells.append(f"{v}: E1M " + ", ".join(f"({c}) {'neg' if ok else 'NOT'} {b:.2f}" for c, (ok, b) in m_cells))
            elif v != "original" and name in ("concrete", "wine"):
                cells.append(f"{v}: E1M VOID (rate tolerance)")
        print(f"  {name}:")
        for c in cells:
            print(f"      {c}")

    print("\n(b) mixing arms vs local-only at t2 - survives if every decision is unchanged")
    for name in DATASETS:
        print(f"  {name}:")
        decisions = {}
        for v, _, _ in VERSIONS:
            e1 = data[v][0]
            if name not in e1:
                continue
            t = rmse(e1[name][1]["metrics"], "t2")
            decisions[v] = {}
            for arm in MIXING:
                c = ci(t, arm, "local-only")
                decisions[v][arm] = decide(c["lo"], c["hi"])
                decisions[v][arm + "_txt"] = f"{c['mean_pct']:+6.2f} [{c['lo']:+6.2f},{c['hi']:+6.2f}]"
        for arm in MIXING:
            row = "   ".join(f"{v} {decisions[v][arm + '_txt']} {decisions[v][arm][:10]:10s}" for v in decisions)
            print(f"      {arm:23s} {row}")
        for v in decisions:
            if v != "original":
                same = all(decisions[v][a] == decisions["original"][a] for a in MIXING)
                print(f"      -> {v}: (b) {'SURVIVES' if same else 'DOES NOT SURVIVE'} "
                      f"({sum(decisions[v][a] == decisions['original'][a] for a in MIXING)}/{len(MIXING)} unchanged)")

    print("\n(b') E1M mixing vs local-only at t2 (kin8nm, protein; original vs corrected)")
    for name in ["kin8nm", "protein"]:
        for cond in "abc":
            parts = []
            for v, _, _ in VERSIONS:
                e1m = data[v][1]
                if (name, cond) in e1m:
                    t = rmse(e1m[(name, cond)][1]["metrics"], "t2")
                    c = ci(t, "uniform-donor", "local-only")
                    parts.append(f"{v} uniform vs local {c['mean_pct']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}] "
                                 f"{decide(c['lo'], c['hi'])}")
            print(f"  {name} ({cond}): " + " | ".join(parts))

    print("\n(5) functional comparison (corrected runs; RMS prediction difference on the receiver's test inputs, "
          "relative to its local-only test RMSE)")
    for v, _, _ in VERSIONS[1:]:
        e1 = data[v][0]
        for name in DATASETS:
            if name not in e1 or "function" not in e1[name][1]:
                continue
            f = e1[name][1]["function"]
            cc = f[f.kind == "client-client"].rel
            arm = f[(f.kind == "arm") & (f.b == "uniform-donor")]
            sing = f[f.kind == "single"]
            a_t = arm.groupby(["a", "timepoint"]).rel.mean().unstack()
            s_t = sing.groupby(["b", "timepoint"]).rel.mean().unstack()
            print(f"  {v:17s} {name:9s} client-client {cc.mean():.3f} (range {cc.min():.3f}-{cc.max():.3f})")
            print("      arm vs uniform mix (t1 / t2): " + "; ".join(
                f"{a} {r.t1:.3f}/{r.t2:.3f}" for a, r in a_t.iterrows()))
            print("      single-donor mix vs (t1 / t2): " + "; ".join(
                f"{b} {r.t1:.3f}/{r.t2:.3f}" for b, r in s_t.iterrows()))

    print("\n(secondary) parameter geometry, corrected runs")
    for v, _, _ in VERSIONS[1:]:
        e1 = data[v][0]
        for name in DATASETS:
            if name not in e1 or "geometry" not in e1[name][1]:
                continue
            g = e1[name][1]["geometry"]
            cc = g[g.kind == "client-client"].dist.mean() / g[g.kind == "client-theta0"].dist.mean()
            arm = g[(g.kind == "arm") & (g.a.isin(SCORE_ARMS))].to_uniform_mix
            sing = g[g.kind == "single"].to_centroid
            print(f"  {v:17s} {name:9s} client-client / client-theta0 {cc:.2f}; score-arm mix to uniform mix "
                  f"{arm.min():.3f}-{arm.max():.3f}; single-donor mix to centroid {sing.mean():.3f}")


if __name__ == "__main__":
    main()
