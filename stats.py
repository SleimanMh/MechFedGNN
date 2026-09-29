"""Paired-difference uncertainty and E1 companion diagnostics (CLAUDE.md §8, §8.1).

Effect unit: relative error in percent of the comparator's RMSE,
  delta_pct = 100 * (RMSE_arm - RMSE_comparator) / RMSE_comparator   (negative = arm better).
Unit of replication: the seed. Receivers within a seed share data, donors and
theta_0, so delta_pct is averaged over receivers within each seed first; the
estimate is the mean over seeds with a 95% t-interval (df = seeds - 1).
Datasets are never pooled. Decision against the fixed threshold DELTA_PCT:
meaningful if the whole interval lies beyond +-delta, negligible if inside it,
unresolved otherwise.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import t as student_t
from sklearn.metrics import adjusted_rand_score

from loop import ARMS, ARMS_ORDER
from scores import missingness_similarity, rate_similarity
from signatures import population_similarity

DELTA_PCT = 2.0
SCORE_ARMS = ARMS[3:]
CONTRASTS = ([(a, "local-only") for a in ARMS[1:]] + [(a, "fedavg") for a in ARMS[2:]]
             + [(a, "uniform-donor") for a in SCORE_ARMS])


def _t(n):
    return student_t.ppf(0.975, n - 1) if n > 1 else float("nan")


def rmse_table(m, fold, timepoint):
    """(seed, receiver) x arm RMSE for one fold and timepoint."""
    r = m[(m.metric == "rmse") & (m.fold == fold) & (m.timepoint == timepoint)]
    return r.pivot_table(index=["seed", "receiver"], columns="arm", values="value")


def paired(tab, arm, comparator):
    """Per (seed, receiver) relative difference in percent."""
    return 100.0 * (tab[arm] - tab[comparator]) / tab[comparator]


def seed_interval(d):
    """Mean over seeds of the within-seed receiver mean, with a 95% t-interval."""
    per_seed = d.groupby(level="seed").mean()
    n = len(per_seed)
    mean, sd = float(per_seed.mean()), float(per_seed.std(ddof=1)) if n > 1 else float("nan")
    hw = _t(n) * sd / np.sqrt(n)
    return {"mean_pct": mean, "lo": mean - hw, "hi": mean + hw, "sd_seed": sd, "seeds": n}


def decide(lo, hi, delta=DELTA_PCT):
    if not np.isfinite(lo):
        return "unresolved"
    if hi < -delta:
        return "meaningful (arm better)"
    if lo > delta:
        return "meaningful (arm worse)"
    if -delta < lo and hi < delta:
        return "negligible"
    return "unresolved"


def contrast_table(m, fold="test", timepoint="t2", delta=DELTA_PCT):
    tab = rmse_table(m, fold, timepoint)
    present = set(tab.columns)
    rows = []
    for arm, comp in [(a, b) for a, b in CONTRASTS if a in present and b in present]:
        ci = seed_interval(paired(tab, arm, comp))
        rows.append({"arm": arm, "vs": comp, **ci, "decision": decide(ci["lo"], ci["hi"], delta)})
    return pd.DataFrame(rows)


def power_table(m, n_study, fold="val", timepoint="t2", delta=DELTA_PCT):
    """Predicted 95% half-width at n_study seeds from the pilot's seed-level SD;
    resolvable if hw <= delta / 2 (§8.1)."""
    out = contrast_table(m, fold, timepoint, delta)[["arm", "vs", "mean_pct", "sd_seed", "seeds"]]
    out["hw_at_study"] = _t(n_study) * out["sd_seed"] / np.sqrt(n_study)
    out["resolvable"] = out["hw_at_study"] <= delta / 2
    return out


def selected_candidate(m, cand):
    """Per (seed, receiver): among local-only and the single-donor candidates, the
    one with the lowest VALIDATION RMSE at t2; returns its TEST RMSE with
    local-only and fedavg test RMSE beside it."""
    def t2(df, fold, key):
        r = df[(df.metric == "rmse") & (df.fold == fold) & (df.timepoint == "t2")]
        return r.pivot_table(index=["seed", "receiver"], columns=key, values="value")
    val = t2(cand, "val", "donor").join(t2(m, "val", "arm")[["local-only"]])
    test = t2(cand, "test", "donor").join(t2(m, "test", "arm")[["local-only", "fedavg"]])
    pick = val.idxmin(axis=1)
    sel = pd.Series([test.loc[idx, c] for idx, c in pick.items()], index=pick.index)
    return pd.DataFrame({"picked": pick, "selected": sel, "local-only": test["local-only"],
                         "fedavg": test["fedavg"]})


def group_recovery(params_by_seed, groups_key="group"):
    """ARI of a 2-cluster average-linkage split on 1 - similarity, per symmetric score."""
    fns = {"rate": lambda a, b: rate_similarity(a["sig"]["r"], b["sig"]["r"])[0],
           "s": lambda a, b: missingness_similarity(a["sig"]["C"], b["sig"]["C"])[0],
           "S": lambda a, b: population_similarity(a["hist"], b["hist"])}
    rows = []
    for seed, params in params_by_seed.items():
        ids = sorted(params)
        truth = [params[c][groups_key] for c in ids]
        for name, fn in fns.items():
            sim = np.array([[1.0 if a == b else (fn(params[a], params[b]) or 0.0) for b in ids] for a in ids])
            dist = np.clip(1.0 - sim, 0, None)
            np.fill_diagonal(dist, 0.0)
            labels = fcluster(linkage(squareform(dist, checks=False), "average"), 2, "maxclust")
            rows.append({"seed": seed, "score": name, "ari": adjusted_rand_score(truth, labels)})
    return pd.DataFrame(rows).groupby("score")["ari"].agg(["mean", "min", "max"])


def compute_by_arm(comp, K, arms=None):
    """Mean optimiser steps / examples / seconds per receiver-seed, per arm.
    local-only: receiver's local training + adaptation. Mixing arms: the local
    training of every client mixed (K - 1 full donors + the receiver) + adaptation."""
    c = comp.copy()
    arms = arms or ARMS
    donors = c[c.receiver == "-"].groupby("seed")[["steps", "examples", "seconds"]].sum() * (K - 1) / K
    rows = []
    for arm in arms:
        a = c[c.component == f"adapt:{arm}"].set_index(["seed", "receiver"])[["steps", "examples", "seconds"]]
        own = c[c.component.str.startswith("local:") & (c.receiver != "-")].set_index(
            ["seed", "receiver"])[["steps", "examples", "seconds"]]
        tot = a + own
        if arm != "local-only":
            tot = tot.add(donors.reindex(tot.index.get_level_values("seed")).set_axis(tot.index))
        rows.append({"model": arm, **tot.mean().to_dict()})
    for ref in ["headroom:local", "headroom:pooled"]:
        r = c[c.component == ref][["steps", "examples"]].mean()
        rows.append({"model": ref, **r.to_dict(), "seconds": float("nan")})
    both = c[c.component == "headroom:both"]["seconds"].mean()
    rows.append({"model": "headroom (both refs, wall-clock)", "steps": float("nan"),
                 "examples": float("nan"), "seconds": both})
    return pd.DataFrame(rows)
