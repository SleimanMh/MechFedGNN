"""E1M construction checks (CLAUDE.md §10) - run BEFORE anything trains.

Verifications, not targets: rate vectors against the declared tolerance,
pi_A / pi_B correlation matching, within-panel phi per group, s-based group
recovery. If the rate tolerance fails, the construction is void.

Run:  python check_e1m.py > results/e1m/checks.txt
"""
import itertools

import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from data.design import load_dataset
from data.inject import load_or_build_roles
from data.matched import FOLDS, build_matched_clients
from scores import missingness_similarity, rate_similarity
from signatures import phi

TOL = 0.01


def s_ari(Ms, groups):
    Cs = [phi(M) for M in Ms]
    n = len(Cs)
    sim = np.array([[1.0 if i == j else (missingness_similarity(Cs[i], Cs[j])[0] or 0.0)
                     for j in range(n)] for i in range(n)])
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    return adjusted_rand_score(groups, fcluster(linkage(squareform(dist, checks=False), "average"), 2, "maxclust"))


def main():
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    verdicts = []
    for name in cfg["datasets"]:
        df, _ = load_dataset(name, verbose=False)
        roles = load_or_build_roles(name, df)
        feats, mk = roles["features"], roles["maskable"]
        _, _, info = build_matched_clients(df, roles, cfg, cfg["seeds"][0], "b")
        q = info["quality"]
        nm = lambda pi: [f"{feats[a]}-{feats[b]}" for a, b in pi]
        print(f"\n######## {name}  maskable {[feats[f] for f in mk]}")
        print(f"  pi_A {nm(info['pi_A'])}  pi_B {nm(info['pi_B'])}  singleton "
              f"{feats[info['singleton']] if info['singleton'] is not None else '-'}")
        print(f"  correlation matching (design rows): mean within-pair |corr| A {q['mean_corr_A']:.4f}  "
              f"B {q['mean_corr_B']:.4f}  |diff| {q['abs_diff']:.4f}  (candidates {q['n_candidates']})")
        print(f"  ordering p {info['p']:.4f}; expected per-feature rate {info['expected_rate']:.4f}; tolerance {TOL}")
        for cond, label in [("a", "(a) cell, null"), ("b", "(b) panel, independent"), ("c", "(c) panel, coupled")]:
            worst, phis, aris, rsims = 0.0, {0: {"A": [], "B": []}, 1: {"A": [], "B": []}}, [], []
            first = None
            for seed in cfg["seeds"]:
                clients, groups, info = build_matched_clients(df, roles, cfg, seed, cond)
                for fold in FOLDS:
                    R = np.array([1 - c["M"][c[fold]][:, mk].mean(0) for c in clients])
                    worst = max(worst, float(R.max() - R.min()))
                    if seed == cfg["seeds"][0] and fold == "train":
                        first = (R, [len(c["train"]) for c in clients])
                trains = [c["M"][c["train"]] for c in clients]
                for c, M in zip(clients, trains):
                    C = phi(M)
                    phis[c["group"]]["A"].append(np.mean([C[a, b] for a, b in info["pi_A"]]))
                    phis[c["group"]]["B"].append(np.mean([C[a, b] for a, b in info["pi_B"]]))
                aris.append(s_ari(trains, list(groups)))
                rs = [rate_similarity(1 - trains[i].mean(0), 1 - trains[j].mean(0))[0]
                      for i, j in itertools.combinations(range(len(trains)), 2)]
                rsims += rs
            ok = worst <= TOL
            verdicts.append((name, label, ok, worst))
            R, sizes = first
            print(f"  {label}")
            print(f"    max |r_i,f - r_j,f| over all folds, seeds, clients, features: {worst:.4f} -> "
                  f"{'MATCHED' if ok else 'NOT MATCHED - construction void'}")
            print(f"    seed {cfg['seeds'][0]} train-fold rates (n_train {sizes}):")
            for k, row in enumerate(R):
                print(f"      c{k} (group {'AB'[clients[k]['group']]}) " + " ".join(
                    f"{feats[f]}={v:.4f}" for f, v in zip(mk, row)))
            print(f"    rate similarity across all client pairs: min {min(rsims):.6f} max {max(rsims):.6f}")
            print(f"    within-panel phi (train, mean over clients x seeds):  "
                  f"group A on pi_A {np.mean(phis[0]['A']):+.3f}, on pi_B {np.mean(phis[0]['B']):+.3f} | "
                  f"group B on pi_A {np.mean(phis[1]['A']):+.3f}, on pi_B {np.mean(phis[1]['B']):+.3f}")
            print(f"    s-based group recovery ARI over seeds: mean {np.mean(aris):.3f} "
                  f"min {np.min(aris):.3f} max {np.max(aris):.3f}")
    print("\nRATE TOLERANCE SUMMARY")
    for name, label, ok, worst in verdicts:
        print(f"  {name:9s} {label:24s} {worst:.4f} {'MATCHED' if ok else 'VOID'}")


if __name__ == "__main__":
    main()
