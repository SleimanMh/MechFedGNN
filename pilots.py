"""Stage 2 pilots (CLAUDE.md §12): adaptation budget, then receiver_fraction.

Both on seed 11 and the VALIDATION fold only; the test fold is untouched.
Selection rules are fixed here, before any result is seen:

Budget: candidates {10, 25, 50} (0 means no adaptation, so timepoint 2 would
equal timepoint 1; 100 has no larger neighbour to check stability against).
  spread(b)    = median over receivers of (max - min val RMSE over the 8
                 collaborative arms) / local-only val RMSE
  stability(b) = mean over receivers of Kendall tau between the arm orderings
                 at b and at the next larger budget
  Choose the smallest b with spread >= 0.01 AND stability >= 0.6 on EVERY dataset.
  If none qualifies, choose the b with the highest minimum stability and flag it.

receiver_fraction over {0.2, 0.3, 0.5} at the chosen budget:
  Choose the largest f where every receiver on every dataset is HEADROOM OK.
  If none, the largest f with no NO HEADROOM anywhere, flagged. Else 0.2, flagged.
"""
import copy
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

from data.design import load_dataset
from data.inject import load_or_build_roles
from headroom import format_block
from loop import ARMS, run_seed

SEED = 11
BUDGETS = [0, 10, 25, 50, 100]
CANDIDATE_BUDGETS = [10, 25, 50]
FRACTIONS = [0.2, 0.3, 0.5]
COLLAB = [a for a in ARMS if a != "local-only"]
OUT = "results/pilots"


def budget_table(metrics):
    m = metrics[metrics.metric == "rmse"]
    rows = []
    for b_i, b in enumerate(BUDGETS):
        spreads, taus = [], []
        for rec, g in m.groupby("receiver"):
            at = g[g.budget == b].set_index("arm")["value"]
            spreads.append((at[COLLAB].max() - at[COLLAB].min()) / at["local-only"])
            if b_i + 1 < len(BUDGETS):
                nxt = g[g.budget == BUDGETS[b_i + 1]].set_index("arm")["value"]
                taus.append(kendalltau(at[ARMS], nxt[ARMS]).statistic)
        at_b = m[m.budget == b].groupby("arm")["value"].mean()
        rows.append({"budget": b, "spread": np.median(spreads),
                     "stability": np.mean(taus) if taus else np.nan,
                     "local": at_b["local-only"], "fedavg": at_b["fedavg"],
                     "uniform": at_b["uniform-donor"], "best_score_arm": at_b[COLLAB[2:]].min()})
    return pd.DataFrame(rows)


def choose_budget(tables):
    ok = [b for b in CANDIDATE_BUDGETS
          if all((t.set_index("budget").loc[b, "spread"] >= 0.01)
                 and (t.set_index("budget").loc[b, "stability"] >= 0.6) for t in tables.values())]
    if ok:
        b = ok[0]
        return b, (f"smallest budget in {CANDIDATE_BUDGETS} with spread >= 0.01 and stability >= 0.6 "
                   f"on every dataset (qualifying: {ok})")
    best = max(CANDIDATE_BUDGETS,
               key=lambda b: min(t.set_index("budget").loc[b, "stability"] for t in tables.values()))
    return best, f"FLAG: no budget met both thresholds on every dataset; chose highest min-stability {best}"


def choose_fraction(verdicts):
    def all_are(f, allowed):
        return all(v in allowed for v in verdicts[f])
    ok = [f for f in FRACTIONS if all_are(f, {"HEADROOM OK"})]
    if ok:
        return max(ok), f"largest fraction with HEADROOM OK for every receiver on every dataset (qualifying: {ok})"
    ok = [f for f in FRACTIONS if all_are(f, {"HEADROOM OK", "THIN HEADROOM"})]
    if ok:
        return max(ok), f"FLAG: no fraction was OK everywhere; largest with no NO HEADROOM (qualifying: {ok})"
    return 0.2, "FLAG: every fraction had a NO HEADROOM receiver; fell back to 0.2"


def freeze(budget, fraction, path="configs/defaults.yaml"):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"adapt_budget: \d+ +# PILOT.*", f"adapt_budget: {budget}                           # PILOT - chosen by pilots.py", text)
    text = re.sub(r"receiver_fraction: [\d.]+ +# PILOT.*", f"receiver_fraction: {fraction}                     # PILOT - chosen by pilots.py", text)
    open(path, "w", encoding="utf-8").write(text)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    data = {}
    for name in cfg["datasets"]:
        df, _ = load_dataset(name)
        data[name] = (df, load_or_build_roles(name, df))
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    say(f"=== BUDGET PILOT  seed={SEED}  fold=val  receiver_fraction={cfg['clients']['receiver_fraction']}")
    tables = {}
    for name, (df, roles) in data.items():
        out = run_seed(df, roles, cfg, SEED, split="val", budgets=BUDGETS)
        tables[name] = budget_table(pd.DataFrame(out["metrics"]))
        say(f"\n{name}  (val RMSE: mean over receivers)")
        say(tables[name].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    budget, why_b = choose_budget(tables)
    say(f"\nCHOSEN adapt_budget = {budget}: {why_b}")

    cfg_b = copy.deepcopy(cfg)
    cfg_b["model"]["adapt_budget"] = budget
    say(f"\n=== RECEIVER_FRACTION PILOT  seed={SEED}  fold=val  adapt_budget={budget}")
    verdicts, blocks, summary = {f: [] for f in FRACTIONS}, {}, []
    for f in FRACTIONS:
        cfg_f = copy.deepcopy(cfg_b)
        cfg_f["clients"]["receiver_fraction"] = f
        for name, (df, roles) in data.items():
            rows = run_seed(df, roles, cfg_f, SEED, split="val")["headroom"]
            verdicts[f] += [r["verdict"] for r in rows]
            blocks[(f, name)] = format_block(rows)
            v = [r["verdict"] for r in rows]
            summary.append({"fraction": f, "dataset": name, "n_train_receiver": rows[0]["n_train"],
                            "OK": v.count("HEADROOM OK"), "THIN": v.count("THIN HEADROOM"),
                            "NONE": v.count("NO HEADROOM"),
                            "median_rel": np.median([r["rel_headroom"] for r in rows])})
    say(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fraction, why_f = choose_fraction(verdicts)
    say(f"\nCHOSEN receiver_fraction = {fraction}: {why_f}")

    say(f"\n=== HEADROOM PER CLIENT at receiver_fraction={fraction}, adapt_budget={budget} (seed {SEED}, val fold)")
    for name in data:
        say(f"\n{name}\n{blocks[(fraction, name)]}")
    freeze(budget, fraction)
    say("\nfrozen in configs/defaults.yaml")
    open(os.path.join(OUT, "pilots.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
