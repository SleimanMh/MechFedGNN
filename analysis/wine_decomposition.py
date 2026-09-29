"""EXPLORATORY: decompose wine's change into duplicate grouping, shared scaler
and the f7/f2 role flip.

The role flip is not an independent factor - it is a downstream consequence of
duplicate grouping changing the design split, which changes the |corr(f,target)|
ranking on the design rows. So it is isolated by running the corrections with
E1's roles forced, and comparing against the corrected roles.

Factorial (headroom skipped; it does not enter any metric):
  A  dup 0  scaler 0  roles E1         = original E1
  B  dup 1  scaler 0  roles E1
  C  dup 0  scaler 1  roles E1
  D  dup 1  scaler 1  roles E1         = corrections without the role flip
  E  dup 1  scaler 1  roles corrected  = corrected-corr as run

Reported: mixing vs local-only at t2 (the §16 (b) criterion) and score arms vs
uniform-donor (the (a) criterion).

Run:  python -m analysis.wine_decomposition > results/wine_decomposition.txt
"""
import copy

import pandas as pd
import yaml

from data.design import duplicate_groups, load_dataset
from data.inject import load_or_build_roles
from loop import run_seed
from stats import DELTA_PCT, decide, seed_interval

CELLS = [("A original", 0, 0, "e1"), ("B dup only", 1, 0, "e1"), ("C scaler only", 0, 1, "e1"),
         ("D dup+scaler, E1 roles", 1, 1, "e1"), ("E full corrected", 1, 1, "corrected")]
ARMS_SHOWN = ["fedavg", "uniform-donor", "coverage-W_H", "combined-Q"]


def contrast(m, a, b, tp="t2"):
    r = m[(m.metric == "rmse") & (m.fold == "test") & (m.timepoint == tp)]
    t = r.pivot_table(index=["seed", "receiver"], columns="arm", values="value")
    c = seed_interval(100 * (t[a] - t[b]) / t[b])
    return c, decide(c["lo"], c["hi"])


def main():
    base = yaml.safe_load(open("configs/defaults.yaml"))
    df, _ = load_dataset("wine", verbose=False)
    g = duplicate_groups(df)
    roles = {"e1": load_or_build_roles("wine", df),
             "corrected": load_or_build_roles("wine", df, "configs/roles_corrected/corr", g, "corr")}
    feats = roles["e1"]["features"]
    for k, r in roles.items():
        print(f"{k:10s} maskable {[feats[i] for i in r['maskable']]}")
    print(f"\nwine, test fold, t2; Delta% vs local-only; seed = unit (n={len(base['seeds'])}); "
          f"delta = {DELTA_PCT}%\n")
    rows = []
    for label, dup, scal, which in CELLS:
        cfg = copy.deepcopy(base)
        cfg["corrections"] = {"dup_groups": bool(dup), "shared_scaler": bool(scal)}
        outs = [run_seed(df, roles[which], cfg, s, folds=("val", "test"), headroom=False)
                for s in cfg["seeds"]]
        m = pd.DataFrame([r for o in outs for r in o["metrics"]])
        row = {"cell": label, "dup": dup, "scaler": scal, "roles": which}
        for arm in ARMS_SHOWN:
            c, d = contrast(m, arm, "local-only")
            row[arm] = f"{c['mean_pct']:+6.2f} [{c['lo']:+6.2f},{c['hi']:+6.2f}] {d.split()[0]}"
        c, d = contrast(m, "coverage-W_H", "uniform-donor")
        row["W_H vs uniform"] = f"{c['mean_pct']:+5.2f} [{c['lo']:+5.2f},{c['hi']:+5.2f}] {d.split()[0]}"
        rows.append(row)
        print(f"{label:24s} " + "  ".join(f"{arm}: {row[arm]}" for arm in ARMS_SHOWN))
    t = pd.DataFrame(rows)
    print("\nAttribution of the change in `uniform-donor` vs local-only (t2):")
    get = lambda lab: float(t[t.cell == lab]["uniform-donor"].iloc[0].split()[0])
    print(f"  A original                {get('A original'):+.2f}%")
    print(f"  + duplicate grouping      {get('B dup only') - get('A original'):+.2f}  -> B {get('B dup only'):+.2f}%")
    print(f"  + shared scaler           {get('C scaler only') - get('A original'):+.2f}  -> C {get('C scaler only'):+.2f}%")
    print(f"  both (D, E1 roles)        {get('D dup+scaler, E1 roles') - get('A original'):+.2f}  "
          f"-> D {get('D dup+scaler, E1 roles'):+.2f}%")
    print(f"  + role flip (D -> E)      {get('E full corrected') - get('D dup+scaler, E1 roles'):+.2f}  "
          f"-> E {get('E full corrected'):+.2f}%")
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
