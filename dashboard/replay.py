"""Read completed experiments for read-only replay.

The framework is the source of truth. Nothing here re-implements training,
scoring or aggregation: scores come from saved artifacts, and donor weights are
produced by calling the SAME `loop.arm_weights` the experiments used.

What the saved research runs do and do not contain is recorded in
`availability()` and surfaced in the UI, rather than approximated:

  * realised donor weights were NEVER written by `report.write_run` (only the
    later `runner.execute` writes weights.csv), so for research runs they are
    RECOMPUTED here through the backend policy and labelled as such;
  * there is no event log, no timestamps and no per-step training loss;
  * there is no aggregate squared-error sum or evaluated-record count - only the
    final RMSE - so the RMSE formula panel cannot show actual sum/count;
  * research runs record validation/test METRICS but not validation/test row
    COUNTS (only n_train per client);
  * a research run is ONE federated round (`budget` is adaptation steps, not
    rounds). Multi-round data exists only for the live/demo path.
"""
import glob
import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from loop import ARMS_ORDER, SCORE_OF_ARM, arm_weights
from stats import DELTA_PCT, decide, seed_interval

RESULTS_ROOT = "results"
SCORELESS = ("local-only", "fedavg", "uniform-donor")


def _read(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def list_runs(root: str = RESULTS_ROOT) -> list[dict]:
    """Every completed run directory that has a config and metrics."""
    out = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        if not os.path.exists(os.path.join(d, "metrics.csv")):
            continue
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        exp = os.path.basename(os.path.dirname(d))
        e5 = cfg.get("e5") or {}
        out.append({"run_id": os.path.basename(d), "path": d.replace("\\", "/"),
                    "experiment": exp, "dataset": cfg.get("dataset", "?"),
                    "condition": e5.get("condition") or (cfg.get("e1m") or {}).get("condition"),
                    "s_config": e5.get("s_config"),
                    "seeds": cfg.get("seeds", []),
                    "arms": cfg.get("arms") or ARMS_ORDER})
    return out


@lru_cache(maxsize=32)
def load_run(path: str) -> dict:
    """Load one run into the structure the dashboard renders."""
    cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    metrics = _read(os.path.join(path, "metrics.csv"))
    scores = _read(os.path.join(path, "scores.csv"))
    saved_weights = _read(os.path.join(path, "weights.csv"))
    compute = _read(os.path.join(path, "compute.csv"))
    params = {}
    for p in sorted(glob.glob(os.path.join(path, "params", "*.json"))):
        params[os.path.splitext(os.path.basename(p))[0]] = json.load(open(p, encoding="utf-8"))

    roles = cfg.get("roles") or {}
    feats = roles.get("features") or []
    maskable = roles.get("maskable") or []
    ac = cfg.get("aggregation", {})
    clients = []
    for cid in sorted(params):
        pr = params[cid]
        clients.append({
            "id": cid, "group": pr.get("group"),
            "n_train": pr.get("n_train"),
            "n_val": None, "n_test": None,          # not saved by research runs
            "profile": pr.get("profile"),
            "feature_names": feats,
            "maskable": [feats[i] for i in maskable] if feats else maskable,
            "always_observed": [feats[i] for i in (roles.get("always_observed") or [])] if feats else [],
        })
    return {
        "path": path, "run_id": os.path.basename(path),
        "experiment": os.path.basename(os.path.dirname(path)),
        "dataset": cfg.get("dataset"), "config": cfg,
        "method": {"arms": cfg.get("arms") or ARMS_ORDER,
                   "alpha": ac.get("alpha"), "beta": ac.get("beta"),
                   "gamma": ac.get("gamma"), "lambda_pop": ac.get("lambda_pop"),
                   "local_steps": (cfg.get("model") or {}).get("local_steps"),
                   "adapt_budget": (cfg.get("model") or {}).get("adapt_budget")},
        "seeds": sorted(metrics.seed.unique().tolist()) if metrics is not None else [],
        "clients": clients, "params": params,
        "_metrics": metrics, "_scores": scores, "_saved_weights": saved_weights,
        "_compute": compute,
        "availability": availability(path, metrics, scores, saved_weights, compute),
    }


def availability(path, metrics, scores, saved_weights, compute) -> dict:
    """Explicit inventory: what this run can and cannot show."""
    return {
        "event_log": {"available": False,
                      "note": "research runs record no ordered events or timestamps; "
                              "the timeline below is a snapshot, not an execution trace"},
        "saved_donor_weights": {"available": saved_weights is not None,
                                "note": "report.write_run never wrote weights.csv; donor weights "
                                        "are recomputed by calling loop.arm_weights, the same code "
                                        "path the experiment used"},
        "error_sum_and_count": {"available": False,
                                "note": "only the final RMSE was saved - not the summed squared "
                                        "error or the number of evaluated records"},
        "training_loss_curve": {"available": False,
                                "note": "compute.csv records steps, examples and seconds; "
                                        "per-step loss was not saved"},
        "val_test_row_counts": {"available": False,
                                "note": "only n_train per client was saved"},
        "federated_rounds": {"available": False,
                             "note": "a research run is a single aggregation; `budget` is "
                                     "adaptation steps, not federated rounds"},
        "provenance": {"available": os.path.exists(os.path.join(path, "provenance.json"))},
        "compute": {"available": compute is not None},
    }


def _sample_shares(run: dict) -> tuple[list[str], np.ndarray]:
    ids = [c["id"] for c in run["clients"]]
    n = np.array([c["n_train"] or 0 for c in run["clients"]], float)
    return ids, (n / n.sum() if n.sum() else np.full(len(n), 1.0 / max(len(n), 1)))


def score_table(run: dict, seed: int, receiver: str) -> dict:
    """All saved scores for one receiver, donor by donor."""
    s = run["_scores"]
    if s is None:
        return {}
    g = s[(s.seed == seed) & (s.receiver == receiver)]
    out: dict[str, dict] = {}
    for row in g.itertuples():
        out.setdefault(row.donor, {})[row.score] = None if pd.isna(row.value) else float(row.value)
        out[row.donor]["_U_t1"] = None if pd.isna(row.U_t1) else float(row.U_t1)
        out[row.donor]["_U_t2"] = None if pd.isna(row.U_t2) else float(row.U_t2)
        out[row.donor]["_Q_source"] = getattr(row, "Q_source", None)
    return out


def aggregation_breakdown(run: dict, seed: int, receiver: str, arm: str) -> dict:
    """Step-by-step, using the backend's own weighting policy.

    Returns donor rows plus the intermediate quantities, so the browser renders
    numbers it is given rather than recomputing the method.
    """
    ids, p = _sample_shares(run)
    if receiver not in ids:
        raise KeyError(receiver)
    i = ids.index(receiver)
    ac = run["config"].get("aggregation", {})
    scores = score_table(run, seed, receiver)
    score_key = SCORE_OF_ARM.get(arm)
    per_donor = {j: (scores.get(ids[j], {}).get(score_key) if score_key else None)
                 for j in range(len(ids)) if j != i}

    # arm_weights expects a list indexed by client, each entry a {score_name: value}
    # dict; the receiver's own entry is ignored. Missing keys become None, which
    # the declared fallback already handles.
    every_key = set(SCORE_OF_ARM.values())
    scores_list = [{k: scores.get(cid, {}).get(k) for k in every_key} for cid in ids]
    w, gamma, fallback = arm_weights(arm, i, scores_list, p, ac)
    saved = _saved_weight_lookup(run, seed, receiver, arm)

    rows = []
    for j, cid in enumerate(ids):
        if j == i:
            continue
        dw = None if w is None else float(w[j])
        rows.append({
            "donor": cid,
            "score_name": score_key,
            "score": per_donor[j],
            "score_available": per_donor[j] is not None,
            "sample_share": float(p[j]),
            "donor_weight": dw,
            "effective_contribution": None if dw is None else (1.0 - float(gamma)) * dw,
            "saved_donor_weight": saved.get(cid),
            "U_t2": scores.get(cid, {}).get("_U_t2"),
        })
    eff_sum = sum(r["effective_contribution"] or 0.0 for r in rows)
    total = float(gamma) + eff_sum
    return {
        "receiver": receiver, "arm": arm, "seed": seed,
        "uses_score": score_key is not None,
        "donors": rows,
        "alpha": ac.get("alpha"), "beta": ac.get("beta"),
        "gamma": float(gamma), "lambda_pop": ac.get("lambda_pop"),
        "fallback": fallback or None,
        "fallback_explained": _explain_fallback(fallback),
        "donor_weight_sum": None if w is None else float(np.nansum([r["donor_weight"] for r in rows])),
        "effective_donor_sum": eff_sum,
        "receiver_contribution": float(gamma),
        "total_contribution": total,
        "total_ok": abs(total - 1.0) < 1e-9,
        "weights_source": "saved artifact" if saved else "recomputed via loop.arm_weights",
        "steps": _steps(arm, score_key, rows, p, i, ac, gamma, fallback),
    }


def _saved_weight_lookup(run, seed, receiver, arm) -> dict:
    sw = run.get("_saved_weights")
    if sw is None:
        return {}
    g = sw[(sw.seed == seed) & (sw.receiver == receiver) & (sw.arm == arm)]
    return {r.donor: float(r.weight) for r in g.itertuples()}


def _explain_fallback(reason: str) -> str | None:
    if not reason:
        return None
    return {
        "undefined": "the score was undefined for at least one donor, so it gives no basis for "
                     "ranking them; the declared fallback uses sample-size weighting instead",
        "non-discriminating": "all donor scores were equal to within 1e-9, so the score cannot "
                              "separate donors; the declared fallback uses sample-size weighting",
    }.get(reason, reason)


def _steps(arm, score_key, rows, p, i, ac, gamma, fallback) -> list[dict]:
    """The calculation narrated with this round's actual numbers."""
    alpha, beta = ac.get("alpha"), ac.get("beta")
    steps = []
    if score_key:
        steps.append({"n": 1, "title": "Receiver-to-donor score",
                      "detail": f"score `{score_key}` for each donor, computed from the "
                                f"receiver's and the donor's mask summaries",
                      "values": {r["donor"]: r["score"] for r in rows}})
    else:
        steps.append({"n": 1, "title": "No score used",
                      "detail": f"`{arm}` does not use a receiver-to-donor score",
                      "values": {}})
    steps.append({"n": 2, "title": "Blend with sample-size weighting (alpha)",
                  "detail": f"base_j = alpha*q_j + (1-alpha)*p_j, with alpha = {alpha}"
                            + (" - alpha = 1, so the score is used alone" if alpha == 1 else "")
                            + (f"; FALLBACK ACTIVE ({fallback}): alpha is forced to 0, so base_j = p_j"
                               if fallback else ""),
                  "values": {r["donor"]: r["sample_share"] for r in rows}})
    steps.append({"n": 3, "title": "Sharpening (beta)",
                  "detail": f"base_j^beta with beta = {beta}"
                            + (" - beta = 1, so weights are not sharpened" if beta == 1 else ""),
                  "values": {}})
    steps.append({"n": 4, "title": "Normalise across participating donors",
                  "detail": "w_j = base_j^beta / sum over donors; the receiver itself gets 0",
                  "values": {r["donor"]: r["donor_weight"] for r in rows}})
    steps.append({"n": 5, "title": "Retain the receiver's self-weight (gamma)",
                  "detail": f"gamma = {gamma}: the fraction of the receiver's OWN model kept",
                  "values": {"gamma": float(gamma)}})
    steps.append({"n": 6, "title": "Effective donor contribution",
                  "detail": "(1 - gamma) * w_j; these sum to 1 - gamma, not to 1",
                  "values": {r["donor"]: r["effective_contribution"] for r in rows}})
    steps.append({"n": 7, "title": "Aggregate the parameters",
                  "detail": "theta_new = gamma*theta_receiver + (1-gamma)*sum_j w_j*theta_j",
                  "values": {}})
    return steps


def weight_matrix(run: dict, seed: int, arm: str) -> dict:
    """Rows = receivers, columns = contributors. The diagonal is the self-weight."""
    ids, _ = _sample_shares(run)
    matrix, fallbacks = [], {}
    for rec in ids:
        b = aggregation_breakdown(run, seed, rec, arm)
        row = {d["donor"]: d["effective_contribution"] for d in b["donors"]}
        row[rec] = b["receiver_contribution"]
        matrix.append([row.get(c) for c in ids])
        if b["fallback"]:
            fallbacks[rec] = b["fallback"]
    return {"clients": ids, "matrix": matrix, "arm": arm, "seed": seed,
            "fallbacks": fallbacks,
            "note": "cells are EFFECTIVE contributions: the diagonal is gamma and each row sums to 1"}


def weight_history(run: dict, receiver: str, arm: str) -> dict:
    """Across seeds (a research run has one federated round per seed).

    Weights are often identical across seeds when masks, populations and
    participation do not change. That is reported, not smoothed over.
    """
    seeds = run["seeds"]
    series, raw = {}, {}
    for s in seeds:
        b = aggregation_breakdown(run, s, receiver, arm)
        for d in b["donors"]:
            series.setdefault(d["donor"], []).append(d["donor_weight"])
            raw.setdefault(d["donor"], []).append(d["score"])
    gammas = [aggregation_breakdown(run, s, receiver, arm)["gamma"] for s in seeds]
    flat = all(len({None if v is None else round(v, 12) for v in vals}) == 1
               for vals in series.values()) if series else True
    return {"x": seeds, "x_label": "seed (each run is one federated round)",
            "normalised_weights": series, "raw_scores": raw, "self_weight": gammas,
            "constant": flat,
            "constant_note": ("donor weights are identical across seeds here - with the same masks, "
                              "populations and participation the score inputs do not change"
                              if flat else None)}


# --------------------------------------------------------------------------
# Prediction results
# --------------------------------------------------------------------------

def folds_and_timepoints(run: dict) -> dict:
    m = run["_metrics"]
    if m is None:
        return {"folds": [], "timepoints": [], "budgets": [], "metrics": []}
    return {"folds": sorted(m.fold.unique().tolist()),
            "timepoints": sorted(m.timepoint.unique().tolist()),
            "budgets": sorted(int(b) for b in m.budget.unique()),
            "metrics": sorted(m.metric.unique().tolist())}


def per_receiver_metric(run: dict, fold: str, timepoint: str, metric: str = "rmse") -> dict:
    """Mean over seeds of each (receiver, arm) value, plus the per-seed spread."""
    m = run["_metrics"]
    sel = m[(m.metric == metric) & (m.fold == fold) & (m.timepoint == timepoint)]
    if sel.empty:
        return {"receivers": [], "arms": [], "mean": [], "sd": []}
    arms = [a for a in ARMS_ORDER if a in set(sel.arm)]
    recs = sorted(sel.receiver.unique())
    # reindex explicitly: with a single seed the std pivot is all-NaN and pandas
    # drops those columns, which would otherwise raise on lookup
    mean = sel.pivot_table(index="receiver", columns="arm", values="value",
                           aggfunc="mean").reindex(index=recs, columns=arms)
    sd = sel.pivot_table(index="receiver", columns="arm", values="value",
                         aggfunc="std").reindex(index=recs, columns=arms)
    return {"receivers": recs, "arms": arms, "metric": metric,
            "mean": [[_f(mean.at[r, a]) for a in arms] for r in recs],
            "sd": [[_f(sd.at[r, a]) for a in arms] for r in recs],
            "seeds": int(sel.seed.nunique()),
            "note": "mean over seeds; the seed is the unit of replication"}


def _f(v):
    return None if v is None or pd.isna(v) else float(v)


def contrasts(run: dict, fold: str, timepoint: str, reference: str = "uniform-donor") -> dict:
    """Arm vs reference, using stats.py - the experiments' own procedure.

    delta_pct is averaged over receivers WITHIN a seed, then summarised over
    seeds with a 95% t-interval; the decision is against the pre-declared
    +-2% threshold. Nothing here is recomputed by the browser.
    """
    m = run["_metrics"]
    tab = m[(m.metric == "rmse") & (m.fold == fold) & (m.timepoint == timepoint)] \
        .pivot_table(index=["seed", "receiver"], columns="arm", values="value")
    rows = []
    if reference not in tab.columns:
        return {"reference": reference, "rows": [], "delta_pct": DELTA_PCT,
                "error": f"'{reference}' was not run in this experiment"}
    for arm in [a for a in ARMS_ORDER if a in tab.columns and a != reference]:
        d = 100.0 * (tab[arm] - tab[reference]) / tab[reference]
        ci = seed_interval(d.dropna())
        rows.append({"arm": arm, **{k: _f(v) for k, v in ci.items()},
                     "seeds": ci["seeds"],
                     "decision": decide(ci["lo"], ci["hi"])})
    return {"reference": reference, "fold": fold, "timepoint": timepoint,
            "delta_pct": DELTA_PCT, "rows": rows,
            "note": "negative = the arm has lower RMSE than the reference; "
                    "'negligible' means the whole interval lies inside +-2%"}


def result_explanation(run: dict, fold: str, timepoint: str, receiver: str, arm: str) -> dict:
    """'How was this result computed?' - traced back to saved artifacts only."""
    m = run["_metrics"]
    sel = m[(m.metric == "rmse") & (m.fold == fold) & (m.timepoint == timepoint)
            & (m.receiver == receiver) & (m.arm == arm)]
    per_seed = [{"seed": int(r.seed), "rmse": float(r.value),
                 "fallback": (r.fallback if isinstance(r.fallback, str) and r.fallback else None)}
                for r in sel.itertuples()]
    tp = {"t1": "before local adaptation (the aggregated model as received)",
          "t2": "after local adaptation on the receiver's own training rows"}
    return {
        "receiver": receiver, "arm": arm, "fold": fold, "timepoint": timepoint,
        "per_seed": per_seed,
        "mean_rmse": float(sel.value.mean()) if len(sel) else None,
        "chain": [
            {"step": "Local training",
             "detail": f"{receiver} trains the mask-aware MLP on its OWN rows only; "
                       f"no other client's records are ever transferred"},
            {"step": "Signature exchange",
             "detail": "each client sends only aggregate mask summaries (r, H, J, C) and "
                       "coarse histograms - never rows, labels or per-row masks"},
            {"step": "Weighting",
             "detail": f"the server turns those summaries into donor weights for arm `{arm}` "
                       f"(see the aggregation inspector for this receiver)"},
            {"step": "Aggregation",
             "detail": "theta_new = gamma*theta_receiver + (1-gamma)*sum_j w_j*theta_j"},
            {"step": "Adaptation" if timepoint == "t2" else "No adaptation",
             "detail": tp.get(timepoint, timepoint)},
            {"step": "Scoring",
             "detail": f"RMSE of the resulting model on {receiver}'s held-out {fold} rows"},
        ],
        "unavailable": {
            "squared_error_sum": "not saved - only the final RMSE was written to metrics.csv",
            "evaluated_record_count": "not saved - the per-fold row count was never recorded",
            "per_example_predictions": "deliberately not transferred or stored",
        },
        "caution": "a single seed is one run, not an established finding; the contrast table "
                   "reports the interval over seeds",
    }
