"""E5 - population compatibility under structured missingness (docs/PROTOCOL_E5.md).

One run per (dataset, condition, S-configuration). Conditions P0M0/P0M1/P1M0/P1M1;
S configurations: `primary` (all always-observed characteristics, including the
partition characteristic) and `robust` (partition characteristic excluded).

Run:  python run_e5.py --stage dev  --datasets concrete wine
      python run_e5.py --stage eval --datasets concrete wine kin8nm protein
"""
import argparse
import copy

import yaml

from data.clients import size_check
from data.design import duplicate_groups, load_dataset
from data.e5 import build_e5_clients, partition_column
from data.inject import load_or_build_roles
from data.validate_injector import precondition
from loop import E5_ARMS, for_dataset, run_seed
from report import write_run

CONDITIONS = {"P0M0": (0, 0), "P0M1": (0, 1), "P1M0": (1, 0), "P1M1": (1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--s-config", nargs="+", default=["primary", "robust"],
                    choices=["primary", "robust"])
    ap.add_argument("--stage", default="eval", choices=["dev", "eval"])
    ap.add_argument("--assignment", default="corr")
    ap.add_argument("--headroom", action="store_true", help="also compute the headroom reference")
    args = ap.parse_args()
    base = yaml.safe_load(open("configs/defaults.yaml"))
    seeds = base["e5_plan"]["dev_seeds" if args.stage == "dev" else "eval_seeds"]
    for name in args.datasets or base["datasets"]:
        df, dropped = load_dataset(name)
        g = duplicate_groups(df)
        roles = load_or_build_roles(name, df, f"configs/roles_corrected/{args.assignment}",
                                    g, args.assignment)
        _, sizes = size_check(len(df), base["clients"]["K"], base["clients"]["split"])
        ok, why = precondition(roles, {}, sizes)
        if not ok:
            print(f"[e5] {name}: precondition failed ({'; '.join(why)}) - not run")
            continue
        pcol = partition_column(df, roles, g)
        for cond in args.conditions:
            P, M = CONDITIONS[cond]
            for sconf in args.s_config:
                cfg = copy.deepcopy(for_dataset(base, name))
                cfg["corrections"] = {"dup_groups": True, "shared_scaler": True,
                                      "maskable": args.assignment, "exploratory": True}
                cfg["e5"] = {"P": P, "M": M, "condition": cond, "s_config": sconf,
                             "s_exclude": (pcol,) if sconf == "robust" else (),
                             "partition_characteristic": roles["features"][pcol],
                             "frac_low": base["clients"]["frac_low"] if P else None,
                             "stage": args.stage}
                cfg["arms"] = E5_ARMS
                cfg["seeds"] = seeds
                cfg["model"] = {**cfg["model"], "local_steps": base["e5_plan"]["local_steps"],
                                "adapt_budget": base["e5_plan"]["adapt_budget"]}
                builder = lambda seed, c=cfg: build_e5_clients(df, roles, c, seed)[:2]
                outs = []
                for seed in seeds:
                    outs.append(run_seed(df, roles, cfg, seed, folds=("val", "test"),
                                         headroom=args.headroom, builder=builder))
                    print(f"[e5] {name}/{cond}/{sconf}: seed {seed} done", flush=True)
                exp = f"e5_{args.stage}_{cond}_{sconf}"
                path = write_run(exp, cfg, name, df, roles, dropped, outs)
                print(f"[e5] {name}/{cond}/{sconf}: report -> {path}", flush=True)


if __name__ == "__main__":
    main()
