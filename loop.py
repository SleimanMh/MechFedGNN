"""One-round protocol (CLAUDE.md §8, §9).

Per run seed: one theta_0; every client trains locally (L_pred only); summaries
from training rows; then every client in turn is the receiver (its training
fold shrunk by receiver_fraction), and each arm aggregates, is evaluated
(timepoint 1 = budget 0), adapted on the receiver's training rows and evaluated
again (timepoint 2). local-only gets the same total step count.
"""
import numpy as np

from data.clients import build_clients, shrink_receiver
from headroom import headroom_row, pooled_oracle
from kernel import aggregate, donor_weights
from model import Standardiser, init_params, metrics, predict, train
from scores import combined_q, missingness_similarity, rate_similarity, w_c, w_h
from signatures import histograms, population_similarity, signature

ARMS = ["local-only", "fedavg", "uniform-donor", "marginal-rate", "missingness-similarity",
        "coverage-W_H", "coverage-W_C", "population-S", "combined-Q"]
SCORE_OF_ARM = {"marginal-rate": "rate", "missingness-similarity": "s", "coverage-W_H": "W_H",
                "coverage-W_C": "W_C", "population-S": "S", "combined-Q": "Q"}


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
    """(donor weights, gamma) for one arm; None for local-only."""
    K = len(p)
    if arm == "local-only":
        return None, 1.0
    if arm == "fedavg":
        return donor_weights(i, [None] * K, p, alpha=0.0, beta=1.0), float(p[i])
    if arm == "uniform-donor":
        w = np.full(K, 1.0 / (K - 1))
        w[i] = 0.0
        return w, ac["gamma"]
    q = [None if j == i else scores[j][SCORE_OF_ARM[arm]] for j in range(K)]
    return donor_weights(i, q, p, alpha=ac["alpha"], beta=ac["beta"]), ac["gamma"]


def _fold(c, st, split):
    rows = c[split]
    return st.inputs(c["X"][rows], c["M"][rows]), c["y"][rows]


def run_seed(df, roles, cfg, seed, split="test", budgets=None):
    """Returns dict of row lists: metrics, scores, weights, headroom, plus params."""
    mc, ac = cfg["model"], cfg["aggregation"]
    budgets = sorted(set([0] + list(budgets if budgets is not None else [mc["adapt_budget"]])))
    main_b = budgets[-1]
    clients, groups = build_clients(df, roles, cfg, seed)
    K, d = len(clients), len(roles["features"])
    theta0 = init_params(d, _seed(seed, 10), tuple(mc["hidden"]))

    def fit_local(c, purpose):
        st = Standardiser(c["X"][c["train"]], c["M"][c["train"]], c["y"][c["train"]])
        xin, yt = st.inputs(c["X"][c["train"]], c["M"][c["train"]]), st.target(c["y"][c["train"]])
        return train(theta0, d, xin, yt, mc["local_steps"], _seed(seed, purpose, int(c["id"][1:])), mc), st

    local_full = [fit_local(c, 20)[0] for c in clients]
    summ_full = [summaries(c, roles) for c in clients]
    out = {"metrics": [], "scores": [], "weights": [], "headroom": [],
           "params": {c["id"]: {**summ_full[k], "group": int(groups[k]), "profile": c["profile"]}
                      for k, c in enumerate(clients)}}

    for i, full in enumerate(clients):
        rec = shrink_receiver(full, cfg["clients"]["receiver_fraction"], seed)
        theta_i, st = fit_local(rec, 21)
        view = clients[:i] + [rec] + clients[i + 1:]
        summ = summ_full[:i] + [summaries(rec, roles)] + summ_full[i + 1:]
        sizes = np.array([s["n_train"] for s in summ], float)
        p = sizes / sizes.sum()
        scores = {j: pair_scores(summ[i], summ[j], ac["lambda_pop"]) for j in range(K) if j != i}
        thetas = local_full[:i] + [theta_i] + local_full[i + 1:]
        xin_tr, yt_tr = st.inputs(rec["X"][rec["train"]], rec["M"][rec["train"]]), st.target(rec["y"][rec["train"]])
        xin_ev, y_ev = _fold(rec, st, split)
        adapt_seed = _seed(seed, 30, i)

        def trajectory(theta):
            """RMSE/MAE/AUC at every budget along one adaptation run."""
            res = {}
            def record(step, params):
                res[step] = metrics(y_ev, predict(params, d, xin_ev, st, tuple(mc["hidden"])), st.y_median)
            train(theta, d, xin_tr, yt_tr, main_b, adapt_seed, mc, budgets, record)
            return res

        local_traj = None
        for arm in ARMS:
            w, gamma = arm_weights(arm, i, scores, p, ac)
            theta = theta_i if w is None else aggregate(thetas, i, w, gamma)
            traj = trajectory(theta)
            if arm == "local-only":
                local_traj = traj
            for b, m in traj.items():
                for name in ["rmse", "mae", "auc"]:
                    out["metrics"].append({"seed": seed, "receiver": rec["id"], "arm": arm, "budget": b,
                                           "timepoint": "t1" if b == 0 else "t2", "metric": name,
                                           "value": m[name], "pos_rate": m["pos_rate"]})
            if w is not None:
                for j in range(K):
                    if j != i:
                        out["weights"].append({"seed": seed, "receiver": rec["id"], "arm": arm,
                                               "donor": clients[j]["id"], "weight": float(w[j]),
                                               "gamma": float(gamma)})
        for j, sc in scores.items():
            e = np.zeros(K)
            e[j] = 1.0
            traj = trajectory(aggregate(thetas, i, e, ac["gamma"]))
            U1 = local_traj[0]["rmse"] - traj[0]["rmse"]
            U2 = local_traj[main_b]["rmse"] - traj[main_b]["rmse"]
            for name in ["W_H", "W_C", "s", "rate", "S", "Q"]:
                out["scores"].append({"seed": seed, "receiver": rec["id"], "donor": clients[j]["id"],
                                      "score": name, "value": sc[name], "U_t1": U1, "U_t2": U2,
                                      "Q_source": sc["Q_source"]})
        oracle = pooled_oracle(theta0, d, view, st, cfg, _seed(seed, 40, i))
        loss_pooled = metrics(y_ev, predict(oracle, d, xin_ev, st, tuple(mc["hidden"])), st.y_median)["rmse"]
        out["headroom"].append({"seed": seed, "receiver": rec["id"], "n_train": len(rec["train"]),
                                **headroom_row(local_traj[main_b]["rmse"], loss_pooled, cfg)})
    return out
