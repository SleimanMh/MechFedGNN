"""E1M - matched-marginal test (CLAUDE.md §10, E1M).

One run per (dataset, condition): condition a (cell, null), b (independent
panels, primary), c (coupled panels, secondary). Same seeds, rows, splits,
model, arms and analysis as E1; no receiver asymmetry. Headroom is a reference
only. Parameter geometry is written to geometry.csv.

Run:  python run_e1m.py --datasets wine --conditions b
"""
import argparse
import copy

import yaml

from data.clients import size_check
from data.design import duplicate_groups, load_dataset
from data.inject import load_or_build_roles
from data.matched import CONDITIONS, build_matched_clients
from data.validate_injector import precondition
from loop import for_dataset, run_seed
from report import write_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--corrected", action="store_true", help="§16 correction study (exploratory)")
    ap.add_argument("--assignment", default="corr", choices=["corr", "random"],
                    help="maskable assignment; 'random' only with --corrected")
    ap.add_argument("--conditions", nargs="+", default=["a", "b", "c"], choices=list(CONDITIONS))
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    for name in args.datasets or cfg["datasets"]:
        df, dropped = load_dataset(name)
        if args.corrected:
            cfg["corrections"] = {"dup_groups": True, "shared_scaler": True, "maskable": args.assignment,
                                  "exploratory": True}
            roles = load_or_build_roles(name, df, f"configs/roles_corrected/{args.assignment}",
                                        duplicate_groups(df), args.assignment)
        else:
            roles = load_or_build_roles(name, df)
        _, sizes = size_check(len(df), cfg["clients"]["K"], cfg["clients"]["split"])
        ok, why = precondition(roles, {}, sizes)
        if not ok:
            print(f"[e1m] {name}: precondition failed ({'; '.join(why)}) - not run")
            continue
        feats = roles["features"]
        for cond in args.conditions:
            cfg_d = copy.deepcopy(for_dataset(cfg, name))
            cfg_d["injection"]["mechanism"] = CONDITIONS[cond]
            builder = lambda seed: build_matched_clients(df, roles, cfg_d, seed, cond)[:2]
            _, _, info = build_matched_clients(df, roles, cfg_d, cfg_d["seeds"][0], cond)
            pair = lambda pi: [f"{feats[a]}-{feats[b]}" for a, b in pi]
            cfg_d["e1m"] = {"condition": cond, "mechanism": CONDITIONS[cond], "pi_A": pair(info["pi_A"]),
                            "pi_B": pair(info["pi_B"]),
                            "singleton": feats[info["singleton"]] if info["singleton"] is not None else None,
                            "p": float(info["p"]), "corr_diff": float(info["quality"]["abs_diff"])}
            outs = []
            for seed in cfg_d["seeds"]:
                outs.append(run_seed(df, roles, cfg_d, seed, folds=("val", "test"), builder=builder))
                print(f"[e1m] {name}/{cond}: seed {seed} done", flush=True)
            path = write_run("e1m_corrected_" + args.assignment if args.corrected else "e1m", cfg_d, name, df, roles, dropped, outs)
            print(f"[e1m] {name}/{cond}: report -> {path}", flush=True)


if __name__ == "__main__":
    main()
