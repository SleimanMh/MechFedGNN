"""One-round protocol (CLAUDE.md §8, §9).

Per run seed: one theta_0; every client trains locally (L_pred only); summaries
from training rows; then every client in turn is the receiver (full size, but
ordering its group's rarely-ordered panels even less often - missingness
asymmetry, §4.4), and each arm aggregates, is evaluated
(timepoint 1 = budget 0), adapted on the receiver's training rows and evaluated
again (timepoint 2). local-only gets the same total step count.
"""
import copy
import time

import numpy as np

from data.clients import asymmetric_receiver, build_clients
from headroom import headroom_row, reference_losses
import geometry
from kernel import aggregate, donor_weights
from model import Standardiser, init_params, metrics, predict, train
from scores import combined_q, missingness_similarity, rate_similarity, w_c, w_h
from signatures import histograms, population_similarity, signature

ARMS = ["local-only", "fedavg", "uniform-donor", "marginal-rate", "missingness-similarity",
        "coverage-W_H", "coverage-W_C", "population-S", "combined-Q"]
SCORE_OF_ARM = {"marginal-rate": "rate", "missingness-similarity": "s", "coverage-W_H": "W_H",
                "coverage-W_C": "W_C", "population-S": "S", "combined-Q": "Q"}
NONDISCRIM_TOL = 1e-9   # scores equal within this give no basis for distinguishing donors (§7.1)


def for_dataset(cfg, name):
    """Copy of cfg with cfg['dataset_overrides'][name] deep-merged in."""
    def merge(base, over):
        for k, v in over.items():
            base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) else v
        return base
    return merge(copy.deepcopy(cfg), (cfg.get("dataset_overrides") or {}).get(name, {}))


def _seed(*parts):
    return int(np.random.default_rng(list(parts)).integers(2**31 - 1))


def summaries(c, roles):
    tr = c["train"]
    names = [roles["features"][i] for i in roles["always_observed"]]
    hist = histograms(c["X"][tr][:, roles["always_observed"]], names, roles["bin_edges"])
    return {"sig": signature(c["M"][tr]), "hist": hist, "n_train": len(tr)}


def pair_scores(si, sj, lam_pop):
    """All scores for receiver i <- donor j, with the Q fallback source."""
    WH, _ = w_h(si["sig"]["H"], sj["sig"]["J"])
    WC, _ = w_c(si["sig"]["C"], sj["sig"]["J"])
    s, _ = missingness_similarity(si["sig"]["C"], sj["sig"]["C"])
    rate, _ = rate_similarity(si["sig"]["r"], sj["sig"]["r"])
    S = population_similarity(si["hist"], sj["hist"])
    Q, source = combined_q(WH, S, lam_pop)
    return {"W_H": WH, "W_C": WC, "s": s, "rate": rate, "S": S, "Q": Q, "Q_source": source}


def arm_weights(arm, i, scores, p, ac):
    """(donor weights, gamma, fallback) for one arm; weights None for local-only.

    Declared fallback (CLAUDE.md §7.1): if a score arm's score is undefined for
    ANY donor, or all donors' scores are equal within NONDISCRIM_TOL, the score
    gives no basis for distinguishing donors and the arm uses sample-size
    weighting (the arm's own gamma kept). `fallback` is "" or the reason."""
    K = len(p)
    if arm == "local-only":
        return None, 1.0, ""
    if arm == "fedavg":
        return donor_weights(i, [None] * K, p, alpha=0.0, beta=1.0), float(p[i]), ""
    if arm == "uniform-donor":
        w = np.full(K, 1.0 / (K - 1))
        w[i] = 0.0
        return w, ac["gamma"], ""
    q = [None if j == i else scores[j][SCORE_OF_ARM[arm]] for j in range(K)]
    donors = [q[j] for j in range(K) if j != i]
    if any(v is None for v in donors):
        reason = "undefined"
    elif max(donors) - min(donors) <= NONDISCRIM_TOL:
        reason = "non-discriminating"
    else:
        return donor_weights(i, q, p, alpha=ac["alpha"], beta=ac["beta"]), ac["gamma"], ""
    return donor_weights(i, [None] * K, p, alpha=0.0, beta=1.0), ac["gamma"], reason


def _fold(c, st, split):
    rows = c[split]
    return st.inputs(c["X"][rows], c["M"][rows]), c["y"][rows]


def run_seed(df, roles, cfg, seed, folds=("val", "test"), budgets=None, headroom=True, builder=None):
    """Returns dict of row lists: metrics, candidates, scores, weights, headroom,
    compute, geometry, plus params. `builder(seed) -> (clients, groups)` replaces
    E1's client construction and disables the receiver asymmetry (E1M). Metrics are recorded for every fold in `folds`
    (pilots pass ("val",) so test is never evaluated); U uses the last fold."""
    mc, ac = cfg["model"], cfg["aggregation"]
    budgets = sorted(set([0] + list(budgets if budgets is not None else [mc["adapt_budget"]])))
    main_b = budgets[-1]
    clients, groups = builder(seed) if builder else build_clients(df, roles, cfg, seed)
    K, d = len(clients), len(roles["features"])
    hidden = tuple(mc["hidden"])
    theta0 = init_params(d, _seed(seed, 10), hidden)
    out = {"metrics": [], "candidates": [], "scores": [], "weights": [], "headroom": [], "compute": [],
           "geometry": []}

    def compute(receiver, component, steps, n_rows, seconds):
        out["compute"].append({"seed": seed, "receiver": receiver, "component": component, "steps": int(steps),
                               "examples": int(steps) * min(mc["batch"], n_rows), "seconds": seconds})

    def fit_local(c, purpose, receiver):
        t = time.perf_counter()
        st = Standardiser(c["X"][c["train"]], c["M"][c["train"]], c["y"][c["train"]])
        xin, yt = st.inputs(c["X"][c["train"]], c["M"][c["train"]]), st.target(c["y"][c["train"]])
        theta = train(theta0, d, xin, yt, mc["local_steps"], _seed(seed, purpose, int(c["id"][1:])), mc)
        compute(receiver, f"local:{c['id']}", mc["local_steps"], len(c["train"]), time.perf_counter() - t)
        return theta, st

    local_full = [fit_local(c, 20, "-")[0] for c in clients]
    summ_full = [summaries(c, roles) for c in clients]
    out["params"] = {c["id"]: {**summ_full[k], "group": int(groups[k]), "profile": c.get("profile")}
                     for k, c in enumerate(clients)}
    out["geometry"] += geometry.client_rows(seed, local_full, theta0, [c["id"] for c in clients])

    for i, full in enumerate(clients):
        rec = full if builder else asymmetric_receiver(full, roles, cfg, seed)
        theta_i, st = fit_local(rec, 21, rec["id"])
        view = clients[:i] + [rec] + clients[i + 1:]
        summ = summ_full[:i] + [summaries(rec, roles)] + summ_full[i + 1:]
        sizes = np.array([s["n_train"] for s in summ], float)
        p = sizes / sizes.sum()
        scores = {j: pair_scores(summ[i], summ[j], ac["lambda_pop"]) for j in range(K) if j != i}
        thetas = local_full[:i] + [theta_i] + local_full[i + 1:]
        xin_tr, yt_tr = st.inputs(rec["X"][rec["train"]], rec["M"][rec["train"]]), st.target(rec["y"][rec["train"]])
        evals = {f: _fold(rec, st, f) for f in folds}
        adapt_seed = _seed(seed, 30, i)

        def trajectory(theta, component):
            """{fold: {budget: metrics}} along one adaptation run."""
            res = {f: {} for f in folds}
            def record(step, params):
                for f, (x, y) in evals.items():
                    res[f][step] = metrics(y, predict(params, d, x, st, hidden), st.y_median)
            t = time.perf_counter()
            train(theta, d, xin_tr, yt_tr, main_b, adapt_seed, mc, budgets, record)
            compute(rec["id"], component, main_b, len(rec["train"]), time.perf_counter() - t)
            return res

        def rows(target, name, traj, **extra):
            for f, per_b in traj.items():
                for b, m in per_b.items():
                    for metric in ["rmse", "mae", "auc"]:
                        target.append({"seed": seed, "receiver": rec["id"], name[0]: name[1], "fold": f,
                                       "budget": b, "timepoint": "t1" if b == 0 else "t2", "metric": metric,
                                       "value": m[metric], "pos_rate": m["pos_rate"], **extra})

        local_traj, mixtures, singles = None, {}, {}
        for arm in ARMS:
            w, gamma, fallback = arm_weights(arm, i, scores, p, ac)
            theta = theta_i if w is None else aggregate(thetas, i, w, gamma)
            if w is not None:
                mixtures[arm] = theta
            traj = trajectory(theta, f"adapt:{arm}")
            if arm == "local-only":
                local_traj = traj
            rows(out["metrics"], ("arm", arm), traj, fallback=fallback)
            if w is not None:
                for j in range(K):
                    if j != i:
                        out["weights"].append({"seed": seed, "receiver": rec["id"], "arm": arm,
                                               "donor": clients[j]["id"], "weight": float(w[j]),
                                               "gamma": float(gamma), "fallback": fallback})
        u_fold = folds[-1]
        for j, sc in scores.items():
            e = np.zeros(K)
            e[j] = 1.0
            singles[clients[j]["id"]] = aggregate(thetas, i, e, ac["gamma"])
            traj = trajectory(singles[clients[j]["id"]], f"candidate:{clients[j]['id']}")
            rows(out["candidates"], ("donor", clients[j]["id"]), traj)
            U1 = local_traj[u_fold][0]["rmse"] - traj[u_fold][0]["rmse"]
            U2 = local_traj[u_fold][main_b]["rmse"] - traj[u_fold][main_b]["rmse"]
            for name in ["W_H", "W_C", "s", "rate", "S", "Q"]:
                out["scores"].append({"seed": seed, "receiver": rec["id"], "donor": clients[j]["id"],
                                      "score": name, "value": sc[name], "U_t1": U1, "U_t2": U2,
                                      "U_fold": u_fold, "Q_source": sc["Q_source"],
                                      "same_group": bool(groups[j] == groups[i])})
        out["geometry"] += geometry.mixture_rows(seed, rec["id"], thetas, i, mixtures, singles)
        if headroom:
            t = time.perf_counter()
            ref = reference_losses(theta0, d, rec, view, st, cfg, _seed(seed, 40, i))
            secs = time.perf_counter() - t
            n_pooled = sum(len(c["train"]) for c in view)
            compute(rec["id"], "headroom:local", ref["local_halt"], len(rec["train"]), float("nan"))
            compute(rec["id"], "headroom:pooled", ref["pooled_halt"], n_pooled, float("nan"))
            compute(rec["id"], "headroom:both", 0, 1, secs)
            out["headroom"].append({"seed": seed, "receiver": rec["id"], "n_train": len(rec["train"]),
                                    "rate_maskable": float(1 - rec["M"][rec["train"]][:, roles["maskable"]].mean()),
                                    **headroom_row(ref, cfg)})
    return out
