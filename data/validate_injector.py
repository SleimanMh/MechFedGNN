"""Injector diagnostics (CLAUDE.md §3.5).

Run:  python -m data.validate_injector --datasets concrete housing wine
Writes results/injector_validation/<dataset>.txt
"""
import argparse
import io
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from data.clients import client_pool, group_profiles
from data.inject import inject, load_or_build_roles, panel_probs_for_rate

SEEDS = [11, 23, 37, 53, 71, 89, 101, 113, 131, 149]
SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]


def phi(M):
    """phi between absence indicators; constant columns get 0 (the §5 convention).
    Stage 1 replaces this with signatures.py."""
    A = 1.0 - M.astype(float)
    r = A.mean(0)
    H = A.T @ A / len(A)
    v = r * (1 - r)
    den = np.sqrt(np.outer(v, v))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(den > 0, (H - np.outer(r, r)) / den, 0.0)
    np.fill_diagonal(C, 0.0)
    return C, H, r


def diagnose(M, roles):
    panels = roles["panels"]
    mk = roles["maskable"]
    C, H, r = phi(M)
    within, cross = [], []
    pid = {f: k for k, P in enumerate(panels) for f in P}
    for a in range(len(mk)):
        for b in range(a + 1, len(mk)):
            f, g = mk[a], mk[b]
            (within if pid[f] == pid[g] else cross).append(C[f, g])
    lifts = [H[f, g] / (r[f] * r[g]) for i, f in enumerate(mk) for g in mk[i + 1:]
             if r[f] > 0 and r[g] > 0]
    miss_panels = np.stack([(M[:, P] == 0).all(1) for P in panels], 1).sum(1)
    hist = np.bincount(miss_panels, minlength=len(panels) + 1) / len(M)
    return {
        "rate_maskable": 1 - M[:, mk].mean(),
        "rate_all": 1 - M.mean(),
        "r": r,
        "within": np.mean(within) if within else np.nan,
        "cross": np.mean(cross) if cross else np.nan,
        "lift_mean": np.mean(lifts), "lift_max": np.max(lifts),
        "panels_hist": hist,
    }


def run_config(X, y, roles, rate, jitter, mechanism, **kw):
    p = panel_probs_for_rate(len(roles["panels"]), rate, jitter)
    out = []
    for s in SEEDS:
        M, _ = inject(X, y, roles, p, mechanism, np.random.default_rng(s), jitter=jitter,
                      driver_seed=s, **kw)
        out.append(diagnose(M, roles))
    return out


def summarise(label, runs):
    f = lambda k: np.nanmean([r[k] for r in runs])
    hist = np.mean([r["panels_hist"] for r in runs], 0)
    extreme = hist[0] + hist[-1]
    flag = "UNREALISTIC" if extreme > 0.9 else "ok"
    return (f"{label:26s} {f('rate_maskable'):6.3f} {f('rate_all'):6.3f} "
            f"{f('within'):7.3f} {f('cross'):7.3f} {f('lift_mean'):6.2f} {f('lift_max'):6.2f}"
            f"   [{' '.join(f'{h:.2f}' for h in hist)}] {flag}")


def validate(name, raw_dir, rate, jitter):
    df = pd.read_csv(os.path.join(raw_dir, f"{name}.csv"))
    roles = load_or_build_roles(name, df)
    pool = client_pool(len(df))
    X = df[roles["features"]].to_numpy(float)[pool]
    y = df["target"].to_numpy(float)[pool]
    n = len(pool)
    feats = roles["features"]
    print(f"=== {name}  n_total={len(df)}  design rows={roles['n_design_rows']}  "
          f"validated on pool n={n}  d={len(feats)}  rate={rate}  jitter={jitter}  seeds={len(SEEDS)}")
    print("|corr(f,target)|:", roles["abs_corr_target"])
    print("maskable       :", [feats[i] for i in roles["maskable"]])
    print("always_observed:", [feats[i] for i in roles["always_observed"]])
    print("constant (never masked, unused):", [feats[i] for i in roles["constant"]])
    prof = group_profiles(len(roles["panels"]))
    print(f"panels: {len(roles['panels'])}   group profile A / B:")
    for k, P in enumerate(roles["panels"]):
        print(f"  P{k}: {str([feats[i] for i in P]):28s} A={prof[0, k]:.1f} B={prof[1, k]:.1f}")
    print(f"  (third-member threshold = median |corr| among maskable = {roles['panel_corr_median']})")

    print("\nconfig                     rate_m rate_a  within   cross  liftmn  liftmax   missing-panels/row [0 1 2 ...]")
    configs = [("mcar", {})]
    configs += [(f"mar o={o}", {"driver_overlap": o}) for o in SWEEP]
    configs += [("fd_mnar top", {"direction": "top"}), ("fd_mnar bottom", {"direction": "bottom"})]
    configs += [(f"cd_mnar s={s}", {"class_spread": s}) for s in SWEEP]
    res = {}
    for label, kw in configs:
        mech = label.split()[0]
        res[label] = run_config(X, y, roles, rate, jitter, mech, **kw)
        print(summarise(label, res[label]))

    print("\nper-feature missing rate (mean over seeds):")
    for label in ["mcar", "mar o=0.5", "fd_mnar top", "cd_mnar s=0.5"]:
        r = np.mean([x["r"] for x in res[label]], 0)
        print(f"  {label:14s}", " ".join(f"{feats[i]}={r[i]:.3f}" for i in roles["maskable"]))

    verdicts = []
    # Null: MCAR cross-panel phi within 3 SE of 0, SE = sqrt((1-p^2)/(n p^2)), p = observed rate.
    cross = np.array([x["cross"] for x in res["mcar"]])
    p_obs = 1 - np.mean([x["rate_maskable"] for x in res["mcar"]])
    se = np.sqrt((1 - p_obs**2) / (n * p_obs**2))
    se_mean = se / np.sqrt(len(cross))
    ok = abs(cross.mean()) <= 3 * se_mean
    verdicts.append(f"NULL (mcar, cross-panel phi): mean={cross.mean():+.4f}  3*SE/sqrt(seeds)="
                    f"{3 * se_mean:.4f}  (per-seed SE={se:.4f}, max|per-seed|={np.abs(cross).max():.4f})"
                    f"  -> {'PASS' if ok else 'FAIL'}")
    for knob, prefix in [("driver_overlap", "mar o="), ("class_spread", "cd_mnar s=")]:
        vals = [np.mean([x["cross"] for x in res[f"{prefix}{v}"]]) for v in SWEEP]
        mono = all(b > a for a, b in zip(vals, vals[1:]))
        verdicts.append(f"DOSE {knob:14s}: " + " -> ".join(f"{v:+.3f}" for v in vals)
                        + f"  -> {'PASS' if mono else 'FAIL'}")
    base = {m: np.mean([x["rate_maskable"] for x in res[lab]])
            for m, lab in [("mcar", "mcar"), ("mar", "mar o=0.5"),
                           ("fd_mnar", "fd_mnar top"), ("cd_mnar", "cd_mnar s=0.5")]}
    spread = max(base.values()) - min(base.values())
    verdicts.append("RATE calibration: " + " ".join(f"{k}={v:.3f}" for k, v in base.items())
                    + f"  spread={spread:.3f} -> {'PASS' if spread <= 0.02 else 'FAIL'}")
    print("\n" + "\n".join(verdicts))
    return roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["concrete", "housing", "wine", "naval"])
    ap.add_argument("--raw", default="./raw")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--out", default="results/injector_validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets:
        buf = io.StringIO()
        with redirect_stdout(buf):
            validate(name, args.raw, args.rate, args.jitter)
        text = buf.getvalue()
        print(text)
        with open(os.path.join(args.out, f"{name}.txt"), "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
