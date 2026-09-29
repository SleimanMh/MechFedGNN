"""E5 Stage B - real-data development run (docs/PROTOCOL_E5.md §9).

Concrete and Wine, DEVELOPMENT seeds only, VALIDATION fold only. Checks
numerical stability, training progress and runtime, and picks the training
budget from learning curves.

Budget rule, declared before the curves are read, and method-agnostic:
  * local_steps  - smallest step count whose mean validation RMSE of the
    LOCAL-ONLY model is within 1 % of that curve's minimum;
  * adapt_budget - smallest step count whose mean validation RMSE of the
    UNIFORM-DONOR control is within 1 % of that curve's minimum.
Neither curve involves any score arm, so the budget cannot be chosen to
maximise separation between methods. The test fold is never touched here.

Run:  python -m analysis.e5_stage_b > results/e5/stage_b.txt
"""
import copy
import time

import numpy as np
import pandas as pd
import yaml

from data.design import duplicate_groups, load_dataset
from data.e5 import build_e5_clients
from data.inject import load_or_build_roles
from loop import E5_ARMS, run_seed

LOCAL_GRID = [10, 20, 30, 50, 100, 200, 300, 500]     # extended after the first run picked the lower bound
ADAPT_GRID = [0, 10, 25, 50, 100, 200, 400]            # extended after the first run picked the upper bound
WITHIN = 0.01


def cfg_for(base, name, P, M, seeds, local_steps, adapt):
    cfg = copy.deepcopy(base)
    cfg["corrections"] = {"dup_groups": True, "shared_scaler": True, "maskable": "corr"}
    cfg["e5"] = {"P": P, "M": M, "condition": f"P{P}M{M}", "s_config": "primary",
                 "s_exclude": (), "partition_characteristic": "-", "frac_low": None, "stage": "B"}
    cfg["arms"] = E5_ARMS
    cfg["seeds"] = seeds
    cfg["model"] = {**base["model"], "local_steps": local_steps, "adapt_budget": adapt}
    return cfg


def val_rmse(m, arm, budget=None):
    r = m[(m.metric == "rmse") & (m.fold == "val") & (m.arm == arm)]
    if budget is not None:
        r = r[r.budget == budget]
    return float(r.value.mean())


def pick(curve):
    """Smallest grid point within WITHIN of the curve minimum."""
    best = min(curve.values())
    return min(k for k, v in curve.items() if v <= best * (1 + WITHIN))


def main():
    base = yaml.safe_load(open("configs/defaults.yaml"))
    seeds = base["e5_plan"]["dev_seeds"]
    data = {}
    for name in ["concrete", "wine"]:
        df, _ = load_dataset(name, verbose=False)
        g = duplicate_groups(df)
        data[name] = (df, load_or_build_roles(name, df, "configs/roles_corrected/corr", g, "corr"))
    print(f"E5 Stage B - development run. Seeds {seeds}; VALIDATION fold only; test never touched.\n")

    print("1) Local-training learning curve (local-only, mean validation RMSE over "
          "clients x seeds; condition P1M1)")
    raw, times = {name: {} for name in data}, {}
    for steps in LOCAL_GRID:
        t0 = time.perf_counter()
        for name, (df, roles) in data.items():
            cfg = cfg_for(base, name, 1, 1, seeds, steps, 0)
            builder = lambda seed, c=cfg: build_e5_clients(df, roles, c, seed)[:2]
            outs = [run_seed(df, roles, cfg, s, folds=("val",), headroom=False, builder=builder)
                    for s in seeds]
            m = pd.DataFrame([r for o in outs for r in o["metrics"]])
            raw[name][steps] = val_rmse(m, "local-only")
            print(f"   steps {steps:5d}  {name:9s} val RMSE {raw[name][steps]:.4f}")
        times[steps] = time.perf_counter() - t0
    # scale-free: each dataset normalised by ITS OWN best over the grid, then averaged
    norm = {k: float(np.mean([raw[n][k] / min(raw[n].values()) for n in raw])) for k in LOCAL_GRID}
    print("   scale-free curve (1.0 = best): " + "  ".join(f"{k}:{v:.4f}" for k, v in norm.items()))
    local_steps = pick(norm)
    print(f"   -> local_steps = {local_steps} (smallest within {WITHIN:.0%} of the minimum); "
          f"wall-clock per grid point {np.round(list(times.values()), 1).tolist()} s")

    print("\n2) Adaptation-budget curve (uniform-donor CONTROL, mean validation RMSE; P1M1)")
    adapt_curve = {}
    for name, (df, roles) in data.items():
        cfg = cfg_for(base, name, 1, 1, seeds, local_steps, max(ADAPT_GRID))
        builder = lambda seed, c=cfg: build_e5_clients(df, roles, c, seed)[:2]
        outs = [run_seed(df, roles, cfg, s, folds=("val",), budgets=ADAPT_GRID, headroom=False,
                         builder=builder) for s in seeds]
        m = pd.DataFrame([r for o in outs for r in o["metrics"]])
        for b in ADAPT_GRID:
            v = val_rmse(m, "uniform-donor", b)
            adapt_curve.setdefault(b, []).append(v)
            print(f"   budget {b:4d}  {name:9s} val RMSE {v:.4f}")
    per_ds = {k: np.array(v) for k, v in adapt_curve.items()}          # {budget: [concrete, wine]}
    best_ds = np.min(np.stack(list(per_ds.values())), axis=0)          # per-dataset best over budgets
    norm_a = {k: float(np.mean(v / best_ds)) for k, v in per_ds.items()}
    print("   scale-free curve (1.0 = best): " + "  ".join(f"{k}:{v:.4f}" for k, v in norm_a.items()))
    adapt = pick(norm_a)
    print(f"   -> adapt_budget = {adapt} (smallest within {WITHIN:.0%} of the minimum)")

    print("\n3) Numerical stability and runtime at the chosen budget")
    for name, (df, roles) in data.items():
        cfg = cfg_for(base, name, 1, 1, seeds, local_steps, adapt)
        builder = lambda seed, c=cfg: build_e5_clients(df, roles, c, seed)[:2]
        t0 = time.perf_counter()
        outs = [run_seed(df, roles, cfg, s, folds=("val",), headroom=False, builder=builder)
                for s in seeds]
        secs = time.perf_counter() - t0
        m = pd.DataFrame([r for o in outs for r in o["metrics"]])
        bad = int(m.value.isna().sum() + np.isinf(m.value).sum())
        w = pd.DataFrame([r for o in outs for r in o["weights"]])
        fb = int((w.fallback != "").sum()) if "fallback" in w else 0
        print(f"   {name:9s} {secs:6.1f}s for {len(seeds)} seeds x 4 receivers x {len(E5_ARMS)} arms; "
              f"non-finite metrics {bad}; fallback rows {fb}; "
              f"val RMSE range {m[m.metric == 'rmse'].value.min():.3f}-"
              f"{m[m.metric == 'rmse'].value.max():.3f}")
    print(f"\nCHOSEN (from validation curves, no score arm involved): local_steps={local_steps}, "
          f"adapt_budget={adapt}")


if __name__ == "__main__":
    main()
