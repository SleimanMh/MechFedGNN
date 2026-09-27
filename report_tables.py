"""Markdown tables for REPORT.md (split out of report.py to keep files small).

Each builder returns a list of markdown lines to append inside one of the six
fixed report sections - no new sections.
"""
import pandas as pd

from loop import ARMS
from stats import (DELTA_PCT, compute_by_arm, contrast_table, decide, group_recovery, paired,
                   seed_interval, selected_candidate)


def _md(df, index=True):
    """Minimal GitHub markdown table (avoids a tabulate dependency)."""
    df = df.reset_index() if index else df
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    head = [str(c) for c in df.columns]
    body = [[fmt(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in body])


def averaging_harm(m):
    """E1 primary question (CLAUDE.md §10): harm from indiscriminate averaging.
    Per receiver: local-only - fedavg and local-only - uniform-donor (RMSE;
    negative = the averaging arm is worse). Per arm: receivers and
    receiver-seeds it harms relative to local-only."""
    r = m[m.metric == "rmse"]
    L = ["**Harm from averaging, per receiver** (RMSE, mean over seeds; negative = worse than local-only)", ""]
    rows = []
    for rec, g in r.groupby("receiver"):
        row = {"receiver": rec}
        for tp in ["t1", "t2"]:
            at = g[g.timepoint == tp].groupby("arm")["value"].mean()
            row[f"local-fedavg {tp}"] = at["local-only"] - at["fedavg"]
            row[f"local-uniform {tp}"] = at["local-only"] - at["uniform-donor"]
        rows.append(row)
    L += [pd.DataFrame(rows).round(4).pipe(_md, index=False), ""]
    counts = []
    for arm in [a for a in ARMS if a != "local-only"]:
        row = {"arm": arm}
        for tp in ["t1", "t2"]:
            g = r[r.timepoint == tp].pivot_table(index=["seed", "receiver"], columns="arm", values="value")
            harmed = g[arm] > g["local-only"]
            per_rec = g.groupby(level="receiver").mean()
            row[f"receivers harmed {tp}"] = f"{int((per_rec[arm] > per_rec['local-only']).sum())}/{len(per_rec)}"
            row[f"receiver-seeds harmed {tp}"] = f"{int(harmed.sum())}/{len(harmed)}"
        counts.append(row)
    L += ["**Receivers harmed relative to local-only, per arm**", "",
          pd.DataFrame(counts).pipe(_md, index=False), ""]
    return L


def contrasts_md(m, fold):
    L = [f"**Paired contrasts** (§8.1): Δ% = 100·(RMSE_arm − RMSE_vs)/RMSE_vs on the {fold} fold; "
         f"mean over seeds of the within-seed receiver mean, 95% t-interval over seeds; "
         f"threshold δ = {DELTA_PCT}% relative RMSE, fixed before this run.", ""]
    for tp in ["t2", "t1"]:
        t = contrast_table(m, fold, tp)
        L += [f"Timepoint {tp}:", "", t.round(3).pipe(_md, index=False), ""]
    return L


def selection_md(m, cand, fold):
    if cand is None or cand.empty or "val" not in set(cand.fold) or fold not in set(cand.fold):
        return ["**Validation-selected candidate**: n/a (needs validation and test folds).", ""]
    sel = selected_candidate(m, cand)
    idx = sel.index
    rows = []
    for comp in ["local-only", "fedavg"]:
        d = pd.Series(100.0 * (sel["selected"] - sel[comp]) / sel[comp], index=idx)
        ci = seed_interval(d)
        rows.append({"selected vs": comp, **ci, "decision": decide(ci["lo"], ci["hi"])})
    picks = sel["picked"].value_counts().to_dict()
    return ["**Validation-selected candidate** (lowest val RMSE at t2 among local-only and the "
            "single-donor mixes; scored on test):", "",
            pd.DataFrame(rows).round(3).pipe(_md, index=False), "",
            f"Picks over receiver-seeds: {picks}", ""]


def fallback_md(w):
    if w.empty or "fallback" not in w:
        return []
    f = w[w.fallback != ""].drop_duplicates(["seed", "receiver", "arm"])
    total = w.drop_duplicates(["seed", "receiver", "arm"]).groupby("arm").size()
    if f.empty:
        return ["Declared fallback (§7.1, sample-size weighting): fired for no arm, receiver or seed.", ""]
    c = f.groupby(["arm", "fallback"]).size().rename("receiver-seeds").reset_index()
    c["of"] = c["arm"].map(total)
    return ["**Declared fallback (§7.1, sample-size weighting) fired:**", "", c.pipe(_md, index=False), ""]


def recovery_md(params_by_seed):
    t = group_recovery(params_by_seed)
    return ["**Group recovery** (2-cluster average linkage on 1 − similarity among the unshifted "
            "clients; ARI against the true latent groups, over seeds):", "", t.round(3).pipe(_md), ""]


def compute_md(comp, K):
    if comp is None or comp.empty:
        return []
    return ["**Compute** (mean per receiver-seed; examples = steps × min(batch, n_train); "
            "a mixing arm includes the local training of every client it mixes). Identical "
            "stopping rules are not equal compute:", "", compute_by_arm(comp, K).round(1).pipe(_md, index=False), ""]
