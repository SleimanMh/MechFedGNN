"""Stage 2 pilots (CLAUDE.md §12), over 3 seeds.

Datasets: the defaults.yaml candidates that pass every precondition.
Receivers keep full size; they differ by missingness asymmetry (§4.4).

Adaptation budget - arms evaluated on the VALIDATION fold. Rule fixed before
any result is seen: candidates {10, 25, 50} (0 = no adaptation, so timepoint 2
would equal timepoint 1; 100 has no larger neighbour for the stability check).
  spread(b)    = median over (seed, receiver) of (max - min val RMSE over the 8
                 collaborative arms) / local-only val RMSE
  stability(b) = mean over (seed, receiver) of Kendall tau between the arm
                 orderings at b and at the next larger budget
  Choose the smallest b with spread >= 0.01 AND stability >= 0.6 on EVERY
  dataset; else the b with the highest minimum stability, flagged.

Headroom - diagnostic only, nothing is chosen from it. Local and pooled
references are early-stopped on the receiver's validation fold and scored on
its TEST fold, as §8 defines headroom.

Run:  python pilots.py [--headroom-only | --power]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

from data.clients import size_check
from data.design import load_dataset
from data.inject import load_or_build_roles
from data.validate_injector import precondition
from headroom import format_block
from loop import ARMS, for_dataset, run_seed
from stats import DELTA_PCT, power_table

SEEDS = [11, 23, 37]
BUDGETS = [0, 10, 25, 50, 100]
CANDIDATE_BUDGETS = [10, 25, 50]
COLLAB = [a for a in ARMS if a != "local-only"]
OUT = "results/pilots"
NL = "\n"


def budget_table(m):
    m = m[m.metric == "rmse"]
    rows = []
    for b_i, b in enumerate(BUDGETS):
        spreads, taus = [], []
        for _, g in m.groupby(["seed", "receiver"]):
            at = g[g.budget == b].set_index("arm")["value"]
            spreads.append((at[COLLAB].max() - at[COLLAB].min()) / at["local-only"])
            if b_i + 1 < len(BUDGETS):
                nxt = g[g.budget == BUDGETS[b_i + 1]].set_index("arm")["value"]
                taus.append(kendalltau(at[ARMS], nxt[ARMS]).statistic)
        mean = m[m.budget == b].groupby("arm")["value"].mean()
        rows.append({"budget": b, "spread": np.median(spreads),
                     "stability": np.mean(taus) if taus else np.nan,
                     "local": mean["local-only"], "fedavg": mean["fedavg"],
                     "uniform": mean["uniform-donor"], "best_score_arm": mean[COLLAB[2:]].min(),
                     "best_score_arm_name": mean[COLLAB[2:]].idxmin()})
    return pd.DataFrame(rows)


def choose_budget(tables):
    get = lambda t, b, col: t.set_index("budget").loc[b, col]
    ok = [b for b in CANDIDATE_BUDGETS
          if all(get(t, b, "spread") >= 0.01 and get(t, b, "stability") >= 0.6 for t in tables.values())]
    if ok:
        return ok[0], (f"smallest budget in {CANDIDATE_BUDGETS} with spread >= 0.01 and stability >= 0.6 "
                       f"on every dataset (qualifying: {ok})")
    best = max(CANDIDATE_BUDGETS, key=lambda b: min(get(t, b, "stability") for t in tables.values()))
    return best, f"FLAG: no budget met both thresholds on every dataset; chose highest min-stability {best}"


def freeze_budget(budget, path="configs/defaults.yaml"):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"adapt_budget: \d+ +# PILOT.*",
                  f"adapt_budget: {budget}                           # PILOT - chosen by pilots.py", text)
    open(path, "w", encoding="utf-8").write(text)


def headroom_table(rows):
    h = pd.DataFrame(rows)
    return h.groupby("receiver").agg(
        n_train=("n_train", "first"), rate=("rate_maskable", "mean"),
        loss_local=("loss_local", "mean"), loss_pooled=("loss_pooled", "mean"),
        rel_mean=("rel_headroom", "mean"), rel_min=("rel_headroom", "min"),
        harm_mean=("pooling_harm", "mean"), n_worse=("pooling_harm", lambda v: int((v > 0).sum())),
        best_local=("local_best", "mean"), halt_local=("local_halt", "mean"),
        best_pooled=("pooled_best", "mean"), halt_pooled=("pooled_halt", "mean"),
        verdicts=("verdict", lambda v: ", ".join(f"{k} x{int(c)}" for k, c in v.value_counts().items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom-only", action="store_true", help="skip the budget pilot")
    ap.add_argument("--power", action="store_true",
                    help="§8.1: seed-level SD of paired contrasts at the frozen budget -> resolvability")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cc = cfg["clients"]
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    say(f"=== CLIENT TRAINING SIZES (K={cc['K']}, precondition >= 100 rows)")
    data = {}
    for name in cfg["datasets"]:
        df, _ = load_dataset(name, verbose=False)
        roles = load_or_build_roles(name, df)
        status, sizes = size_check(len(df), cc["K"], cc["split"])
        ok, why = precondition(roles, {}, sizes)
        say(f"{name:9s} n={len(df):5d} design={roles['n_design_rows']:4d} panels={len(roles['panels'])} "
            f"train/client={sizes} size={status}  -> {'RUN' if ok else 'SKIP: ' + '; '.join(why)}")
        if ok:
            data[name] = (df, roles)

    say(f"{NL}seeds {SEEDS}; receiver_p_rare {cc['receiver_p_rare']} vs group p_rare {cc['p_rare']}")
    budgets = None if (args.headroom_only or args.power) else BUDGETS
    tables, head_rows, power = {}, {}, {}
    for name, (df, roles) in data.items():
        cfg_d = for_dataset(cfg, name)
        outs = [run_seed(df, roles, cfg_d, s, folds=("val",), budgets=budgets,
                         headroom=not args.power) for s in SEEDS]
        m = pd.DataFrame([r for o in outs for r in o["metrics"]])
        if args.power:
            power[name] = power_table(m, n_study=len(cfg["seeds"]))
        elif not args.headroom_only:
            tables[name] = budget_table(m)
        head_rows[name] = [r for o in outs for r in o["headroom"]]

    if args.power:
        n = len(cfg["seeds"])
        say(f"{NL}=== POWER (§8.1): pilot seeds {SEEDS}, val fold, t2, adapt_budget "
            f"{cfg['model']['adapt_budget']}; predicted 95% half-width at {n} seeds; "
            f"resolvable if hw <= delta/2 = {DELTA_PCT / 2}% (delta = {DELTA_PCT}% relative RMSE)")
        for name, t in power.items():
            say(f"{NL}{name}  ({int(t['resolvable'].sum())}/{len(t)} contrasts resolvable){NL}"
                + t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        with open(os.path.join(OUT, "pilots_power.txt"), "w", encoding="utf-8") as f:
            f.write(NL.join(log) + NL)
        return

    say(f"{NL}=== HEADROOM (test fold; local and pooled both early-stopped on val)")
    for name, rows in head_rows.items():
        say(f"{NL}{name}{NL}" + headroom_table(rows).to_string(float_format=lambda v: f"{v:.3f}"))
        say(format_block(rows))
    if args.headroom_only:
        with open(os.path.join(OUT, "pilots_headroom.txt"), "w", encoding="utf-8") as f:
            f.write(NL.join(log) + NL)
        return

    say(f"{NL}=== ADAPTATION BUDGET (val fold; val RMSE means over seeds x receivers)")
    for name, t in tables.items():
        say(f"{NL}{name}{NL}" + t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    budget, why = choose_budget(tables)
    say(f"{NL}CHOSEN adapt_budget = {budget}: {why}")
    freeze_budget(budget)
    say(f"{NL}frozen adapt_budget in configs/defaults.yaml")
    with open(os.path.join(OUT, "pilots.txt"), "w", encoding="utf-8") as f:
        f.write(NL.join(log) + NL)


if __name__ == "__main__":
    main()
