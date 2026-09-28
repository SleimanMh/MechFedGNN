"""E1 - signal comparison (CLAUDE.md §10), the bounded study.

Every dataset in defaults.yaml that passes the §3.5 preconditions runs; none is
excluded for thin or absent headroom (§8: headroom is a reference, not a gate).
All seeds, every client as receiver, all nine arms, validation and test folds
(validation for the selected-candidate diagnostic only). One run directory and
REPORT.md per dataset.

Run:  python run_e1.py [--datasets wine kin8nm]
"""
import argparse

import yaml

from data.clients import size_check
from data.design import duplicate_groups, load_dataset
from data.inject import load_or_build_roles
from data.validate_injector import precondition
from loop import for_dataset, run_seed
from report import write_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--corrected", action="store_true", help="§16 correction study (exploratory)")
    ap.add_argument("--assignment", default="corr", choices=["corr", "random"],
                    help="maskable assignment; 'random' only with --corrected")
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
            print(f"[e1] {name}: precondition failed ({'; '.join(why)}) - not run")
            continue
        cfg_d = for_dataset(cfg, name)
        outs = []
        for seed in cfg_d["seeds"]:
            outs.append(run_seed(df, roles, cfg_d, seed, folds=("val", "test")))
            print(f"[e1] {name}: seed {seed} done", flush=True)
        exp = "e1_corrected_" + args.assignment if args.corrected else "e1"
        path = write_run(exp, cfg_d, name, df, roles, dropped, outs)
        print(f"[e1] {name}: report -> {path}", flush=True)


if __name__ == "__main__":
    main()
