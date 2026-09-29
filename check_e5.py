"""E5 construction checks (docs/PROTOCOL_E5.md §10) - run BEFORE training.

Per dataset, condition, client and fold: sizes and duplicate-group separation,
per-feature rates (against the declared 0.01 tolerance), H, within/between-panel
C, rows with 0/1/many missing features, partial-panel missingness, population
histogram differences, realised W_marg / W_H / S / Q_H / Q_marg, the resulting
weights and any fallbacks.

Run:  python check_e5.py [--datasets wine] [--seeds 2 3 5] > results/e5/checks.txt
"""
import argparse
import copy

import numpy as np
import yaml

from data.design import duplicate_groups, load_dataset
from data.e5 import FOLDS, build_e5_clients, partition_column
from data.inject import load_or_build_roles
from loop import E5_ARMS, arm_weights, pair_scores, summaries
from signatures import phi, population_similarity

CONDITIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def e5_cfg(base, name, P, M, assignment="corr", s_exclude=()):
    cfg = copy.deepcopy(base)
    cfg["corrections"] = {"dup_groups": True, "shared_scaler": True, "maskable": assignment}
    cfg["e5"] = {"P": P, "M": M, "s_exclude": tuple(s_exclude)}
    cfg["arms"] = E5_ARMS
    return cfg


def mask_stats(M, roles):
    mk, panels = roles["maskable"], roles["panels"]
    A = 1 - M[:, mk]
    per_row = A.sum(1)
    C = phi(M)
    pid = {f: k for k, P in enumerate(panels) for f in P}
    win, btw = [], []
    for a in range(len(mk)):
        for b in range(a + 1, len(mk)):
            (win if pid[mk[a]] == pid[mk[b]] else btw).append(C[mk[a], mk[b]])
    partial = 0
    for P in panels:
        if len(P) > 1:
            miss = (M[:, P] == 0).sum(1)
            partial += int(((miss > 0) & (miss < len(P))).sum())
    H = (A.T @ A) / len(A)
    iu = np.triu_indices(len(mk), 1)
    return {"rate": 1 - M[:, mk].mean(0), "H_mean": H[iu].mean() if len(iu[0]) else np.nan,
            "H_max": H[iu].max() if len(iu[0]) else np.nan,
            "within": np.mean(win) if win else np.nan, "between": np.mean(btw) if btw else np.nan,
            "rows0": int((per_row == 0).sum()), "rows1": int((per_row == 1).sum()),
            "rows2p": int((per_row > 1).sum()), "partial_panel_rows": partial, "n": len(M)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--assignment", default="corr")
    args = ap.parse_args()
    base = yaml.safe_load(open("configs/defaults.yaml"))
    seeds = args.seeds or base["e5_plan"]["eval_seeds"]
    verdicts = []
    for name in args.datasets or base["datasets"]:
        df, _ = load_dataset(name, verbose=False)
        g = duplicate_groups(df)
        roles = load_or_build_roles(name, df, f"configs/roles_corrected/{args.assignment}",
                                    g, args.assignment)
        feats = roles["features"]
        pcol = partition_column(df, roles, g)
        print(f"\n######## {name}  n={len(df)}  seeds={seeds}")
        print(f"  partition characteristic (declared, target-free): {feats[pcol]} "
              f"({len(np.unique(df[feats[pcol]]))} distinct values); always_observed "
              f"{[feats[i] for i in roles['always_observed']]}")
        print(f"  maskable {[feats[i] for i in roles['maskable']]}  panels "
              f"{[[feats[i] for i in P] for P in roles['panels']]}")
        for P, M in CONDITIONS:
            cfg = e5_cfg(base, name, P, M, args.assignment)
            cfg_other = e5_cfg(base, name, P, 1 - M, args.assignment)
            worst_rate, sizes, rows_seen, mismatch = 0.0, [], [], 0
            first = None
            for seed in seeds:
                clients, latent, info = build_e5_clients(df, roles, cfg, seed)
                other, _, _ = build_e5_clients(df, roles, cfg_other, seed)
                rows_seen.append(np.concatenate([c["rows"] for c in clients]))
                sizes.append([len(c["rows"]) for c in clients])
                for c, o in zip(clients, other):
                    if not np.array_equal(c["rows"], o["rows"]):
                        mismatch += 1
                    for f in FOLDS:
                        a_cnt = (c["M"][c[f]][:, roles["maskable"]] == 0).sum(0)
                        b_cnt = (o["M"][o[f]][:, roles["maskable"]] == 0).sum(0)
                        mismatch += int(not np.array_equal(a_cnt, b_cnt))
                        R = np.array([1 - x["M"][x[f]][:, roles["maskable"]].mean(0) for x in clients])
                        worst_rate = max(worst_rate, float(R.max() - R.min()))
                if first is None:
                    first = (clients, latent, info)
            clients, latent, info = first
            # duplicate-group separation
            loc = {}
            for c in clients:
                for f in FOLDS:
                    for r in c["rows"][c[f]]:
                        loc.setdefault(g[r], set()).add(f"{c['id']}:{f}")
            split_groups = sum(1 for v in loc.values() if len(v) > 1)
            ok = mismatch == 0 and split_groups == 0
            verdicts.append((name, f"P{P}M{M}", ok, worst_rate, split_groups, mismatch))
            print(f"\n  --- condition P{P}M{M} (seed {seeds[0]} shown; rates over all seeds)")
            print(f"      client sizes {sizes[0]}; duplicate groups split across locations "
                  f"{split_groups} (must be 0); all pool rows used: "
                  f"{len(np.unique(rows_seen[0])) == len(rows_seen[0])}")
            print(f"      M0<->M1 per-feature COUNT mismatches (must be 0, over all "
                  f"seeds/clients/folds): {mismatch} -> {'EXACTLY MATCHED' if not mismatch else 'FAILS'}")
            print(f"      cross-client rate spread within a fold: {worst_rate:.4f} "
                  "(design feature - clients carry different group profiles, not bounded)")
            summ = [summaries(c, roles) for c in clients]
            for k, c in enumerate(clients):
                st = mask_stats(c["M"][c["train"]], roles)
                print(f"      {c['id']} grp{c['group']} n={len(c['rows'])} "
                      f"tr/va/te={len(c['train'])}/{len(c['val'])}/{len(c['test'])} "
                      f"p_vec={np.round(c['profile'], 2).tolist()}")
                print(f"         train rates " + " ".join(
                    f"{feats[f]}={v:.3f}" for f, v in zip(roles["maskable"], st["rate"])))
                print(f"         H mean {st['H_mean']:.3f} max {st['H_max']:.3f}; C within "
                      f"{st['within']:+.3f} between {st['between']:+.3f}; rows 0/1/2+ missing "
                      f"{st['rows0']}/{st['rows1']}/{st['rows2p']}; partial-panel rows "
                      f"{st['partial_panel_rows']}")
            # population separation on the partition characteristic
            med = np.median(np.concatenate([c["X"][:, pcol] for c in clients]))
            print("      population: share of training rows below the pooled median of "
                  f"{feats[pcol]}: " + " ".join(
                      f"{c['id']}={np.mean(c['X'][c['train']][:, pcol] <= med):.2f}" for c in clients))
            Sm = np.array([[population_similarity(summ[i]["hist"], summ[j]["hist"]) or np.nan
                            for j in range(len(clients))] for i in range(len(clients))])
            iu = np.triu_indices(len(clients), 1)
            print(f"      S off-diagonal: min {np.nanmin(Sm[iu]):.4f} max {np.nanmax(Sm[iu]):.4f} "
                  f"spread {np.nanmax(Sm[iu]) - np.nanmin(Sm[iu]):.4f}")
            # realised scores and weights for receiver c0
            K = len(clients)
            p = np.array([s["n_train"] for s in summ], float)
            p /= p.sum()
            sc = {j: pair_scores(summ[0], summ[j], base["aggregation"]["lambda_pop"])
                  for j in range(1, K)}
            print("      receiver c0 scores: " + "; ".join(
                f"{nm}=" + "/".join(f"{sc[j][nm]:.3f}" if sc[j][nm] is not None else "None"
                                    for j in range(1, K))
                for nm in ["W_marg", "W_H", "S", "Q", "Q_marg"]) + "   (donors c1/c2/c3)")
            for arm in E5_ARMS:
                w, gamma, fb = arm_weights(arm, 0, sc, p, base["aggregation"])
                if w is not None:
                    print(f"         {arm:22s} weights " + "/".join(f"{w[j]:.3f}" for j in range(1, K))
                          + f"  gamma {gamma:.2f}" + (f"  FALLBACK: {fb}" if fb else ""))
    print("\nSUMMARY - invariant: M0<->M1 per-feature counts identical, duplicate groups whole")
    for name, cond, ok, worst, split, mm in verdicts:
        print(f"  {name:9s} {cond} count mismatches {mm}; duplicate groups split {split}; "
              f"cross-client rate spread {worst:.4f} (design) -> {'PASS' if ok else 'FAIL'}")
    bad = [f"{n}/{c}" for n, c, ok, _, _, _ in verdicts if not ok]
    print("VERDICT: " + ("all constructions satisfy the design" if not bad
                         else f"FAILING: {bad} - fix the construction before any prediction result"))


if __name__ == "__main__":
    main()
