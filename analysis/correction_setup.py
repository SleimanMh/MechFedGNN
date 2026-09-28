"""§16 correction study - setup report (before any training).

Duplicate rates and rows moved per dataset (vs E1's assignment, all seeds);
roles / panels under both assignments vs E1's; preconditions; E1M rate
tolerance under the corrections.

Run:  python -m analysis.correction_setup > results/correction_setup.txt
"""
import copy

import numpy as np
import yaml

from data.clients import build_clients, size_check
from data.design import design_split, duplicate_groups, load_dataset
from data.inject import load_or_build_roles
from data.matched import FOLDS, build_matched_clients
from data.validate_injector import precondition


def locations(df, roles, cfg, seed, groups):
    loc = np.full(len(df), "design", dtype=object)
    for c in build_clients(df, roles, cfg, seed)[0]:
        for f in FOLDS:
            loc[c["rows"][c[f]]] = f"{c['id']}:{f}"
    return loc


def main():
    base = yaml.safe_load(open("configs/defaults.yaml"))
    for name in base["datasets"]:
        df, _ = load_dataset(name, verbose=False)
        g = duplicate_groups(df)
        _, inv_cnt = np.unique(g, return_counts=True)
        in_group = int((inv_cnt[g] > 1).sum())
        surplus = len(df) - len(inv_cnt)
        orig_roles = load_or_build_roles(name, df)
        feats = orig_roles["features"]
        nm = lambda ix: [feats[i] for i in ix]
        print(f"\n######## {name}  n={len(df)}")
        print(f"  duplicates: {in_group} rows in a duplicate group ({in_group / len(df):.1%}); "
              f"{surplus} surplus copies ({surplus / len(df):.1%}); {int((inv_cnt > 1).sum())} groups")
        cfg_c = copy.deepcopy(base)
        cfg_c["corrections"] = {"dup_groups": True}
        moved = []
        for seed in base["seeds"]:
            a = locations(df, orig_roles, base, seed, None)
            b = locations(df, orig_roles, cfg_c, seed, g)
            moved.append(int((a != b).sum()))
        print(f"  rows moved by duplicate grouping (vs E1's assignment): mean {np.mean(moved):.1f} "
              f"(min {min(moved)}, max {max(moved)}) of {len(df)} per seed")
        print(f"  E1 roles:        maskable {nm(orig_roles['maskable'])}  panels "
              f"{[nm(p) for p in orig_roles['panels']]}")
        for mode in ["corr", "random"]:
            roles = load_or_build_roles(name, df, f"configs/roles_corrected/{mode}", g, mode)
            _, sizes = size_check(len(df), base["clients"]["K"], base["clients"]["split"])
            ok, why = precondition(roles, {}, sizes)
            print(f"  corrected/{mode:6s} maskable {nm(roles['maskable'])}  panels "
                  f"{[nm(p) for p in roles['panels']]}  design rows {roles['n_design_rows']}  "
                  f"precondition {'ACCEPT' if ok else 'REJECT: ' + '; '.join(why)}")
            cfg_m = copy.deepcopy(base)
            cfg_m["corrections"] = {"dup_groups": True, "shared_scaler": True, "maskable": mode}
            worst, sizes_tr = 0.0, set()
            for seed in base["seeds"]:
                clients, _, info = build_matched_clients(df, roles, cfg_m, seed, "b")
                for f in FOLDS:
                    R = np.array([1 - c["M"][c[f]][:, roles["maskable"]].mean(0) for c in clients])
                    worst = max(worst, float(R.max() - R.min()))
                sizes_tr |= {len(c["train"]) for c in clients}
            q = info["quality"]
            print(f"      E1M: pi_A {[nm(p) for p in info['pi_A']]} pi_B {[nm(p) for p in info['pi_B']]} "
                  f"|diff| {q['abs_diff']:.3f}; max rate difference {worst:.4f} "
                  f"({'MATCHED' if worst <= 0.01 else 'VOID'}); training rows per client {sorted(sizes_tr)}")


if __name__ == "__main__":
    main()
