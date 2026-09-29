"""Run artefacts and REPORT.md, per the run-report skill (.claude/skills/run-report).

One run = one dataset x one config x all seeds. Writes, under
results/<exp>/<run_id>/: REPORT.md (six fixed sections), config.json,
metrics.csv, scores.csv, params/<client>.json.
"""
import datetime
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from kernel import aggregate, donor_weights
from loop import ARMS_ORDER
from report_tables import (_jsonable, _md, _pairs, averaging_harm, compute_md, contrasts_md, fallback_md,
                           recovery_md, selection_md)

SCORES = ["W_H", "W_C", "s", "rate", "S", "Q"]
READING = """| Result | Interpretation |
|---|---|
| A score's ordering tracks `U` | That signal carries information about useful transfer |
| No score tracks `U` | The proxies miss what matters in this setting |
| An arm beats local-only and uniform-donor | Its weighting is doing real work |
| An arm beats local-only but not uniform-donor | The gain is from collaborating at all, not from the weights |
| Gains at timepoint 1 but not timepoint 2 | The benefit may be limited to initialisation |
| NO HEADROOM verdict | The pooled reference did not beat local here; a subset-weighted rule still might — read it beside the arm results |"""


def run_id(cfg):
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "nogit"
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{datetime.date.today():%Y%m%d}_{sha}_{h}"


def frames(outs, keys=("metrics", "scores", "weights", "headroom")):
    cat = lambda key: pd.DataFrame([r for o in outs for r in o.get(key, [])])
    return tuple(cat(k) for k in keys)


def write_csvs(out_dir, outs):
    """metrics.csv and scores.csv are deterministic per seed; candidates.csv too.
    compute.csv holds wall-clock seconds, so it is not."""
    os.makedirs(out_dir, exist_ok=True)
    for key in ["metrics", "scores", "candidates", "compute", "geometry", "function"]:
        (t,) = frames(outs, (key,))
        t.to_csv(os.path.join(out_dir, f"{key}.csv"), index=False, float_format="%.10g")


def _mean_sd(g):
    per_seed = g.groupby("seed")["value"].mean()
    return per_seed.mean(), per_seed.std(ddof=0)


def fedavg_limit_ok():
    rng = np.random.default_rng(0)
    p = np.array([0.1, 0.3, 0.6])
    th = [rng.standard_normal(20) for _ in range(3)]
    out = aggregate(th, 1, donor_weights(1, [0.5] * 3, p, 0.0, 1.0), p[1])
    return bool(np.allclose(out, sum(pk * t for pk, t in zip(p, th)), atol=1e-8, rtol=0))


def build_report(cfg, name, df, roles, dropped, outs):
    m_all, s, w, h, cand, comp = frames(outs, ("metrics", "scores", "weights", "headroom",
                                               "candidates", "compute"))
    arms = [a for a in ARMS_ORDER if a in set(m_all.arm)]
    fold = "test" if "test" in set(m_all.fold) else "val"
    e1m = cfg.get("e1m")
    e5c = cfg.get("e5")
    m = m_all[m_all.fold == fold]
    feats, inj, cc, ac, mc = roles["features"], cfg["injection"], cfg["clients"], cfg["aggregation"], cfg["model"]
    first = outs[0]["params"]
    L = [f"# Run report - {cfg.get('experiment', 'run')} / {name}", ""]

    L += ["## 1. Data", "",
          f"- Dataset `{name}`: n = {len(df)}, d = {len(feats)} (exact duplicates dropped: {dropped or 'none'}), "
          "target column `target`; task: regression (MSE) + AUC-ROC of the regression output against "
          "y > receiver training-fold median.",
          f"- Design split: {roles['n_design_rows']} rows (seed {roles['design_seed']}), excluded from all clients.",
          f"- Clients: K = {cc['K']}, rows assigned `{cc['population']}`, disjoint; per-client split "
          f"{cc['split']} (train/val/test) drawn per run seed; seeds {cfg['seeds']}.",
          f"- Training-fold sizes, seed {cfg['seeds'][0]}: "
          + "; ".join(f"{c} {v['n_train']}" for c, v in first.items()),
          (f"- E5 {e5c['condition']} ({'heterogeneous' if e5c['P'] else 'approximately random'} "
           f"populations, {'structured panel' if e5c['M'] else 'independent cell'} masking); "
           f"S configuration `{e5c['s_config']}`, partition characteristic "
           f"{e5c['partition_characteristic']}; no receiver asymmetry."
           if e5c else
           f"- E1M: no receiver asymmetry; rows and splits identical to E1 for each seed."
           if e1m else
           f"- Missingness asymmetry: as receiver, a client orders its group's rarely-ordered panels at "
           f"{cc['receiver_p_rare']} instead of {cc['p_rare']} (mask re-injected, same random draws); "
           "sizes unchanged."),
          f"- Native NaNs before injection: {int(df.isna().sum().sum())} (asserted zero at download).",
          (f"- **§16 correction study (EXPLORATORY)**: {cfg['corrections']}" if cfg.get("corrections") else ""),
          ""]

    mk = [feats[i] for i in roles["maskable"]]
    panel_txt = (f"E1M condition ({e1m['condition']}), mechanism `{e1m['mechanism']}`: group 0 pairs "
                 f"{e1m['pi_A']}, group 1 pairs {e1m['pi_B']}, shared singleton {e1m['singleton']}; every "
                 f"panel ordered with p = {e1m['p']:.4f}; exact counts per fold; correlation-matching "
                 f"|diff| {e1m['corr_diff']:.3f}." if e1m else
                 "Panels: " + ", ".join(f"P{k}={[feats[i] for i in P]}" for k, P in enumerate(roles["panels"])))
    L += ["## 2. Missingness injection", "",
          f"Maskable columns (identical for every client): {mk}. " + panel_txt, "",
          "| client | group | mechanism | panel ordering profile | driver_overlap | class_spread | direction "
          "| realised rate (maskable) | mean lift | mean phi |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in first.items():
        sig, idx = v["sig"], roles["maskable"]
        r, H, C = sig["r"][idx], sig["H"][np.ix_(idx, idx)], sig["C"][np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        rr = np.outer(r, r)[iu]
        lift = np.mean(H[iu][rr > 0] / rr[rr > 0]) if (rr > 0).any() else float("nan")
        L.append(f"| {c} | {v['group']} | {inj['mechanism']} | "
                 f"{np.round(v['profile'], 2).tolist() if v['profile'] is not None else 'matched p'} | "
                 f"{inj['driver_overlap']} | {inj['class_spread']} | {inj['direction']} | "
                 f"{r.mean():.3f} | {lift:.2f} | {C[iu].mean():.3f} |")
    if not e1m:
        L += ["", f"Mechanism `{inj['mechanism']}` with driver_overlap {inj['driver_overlap']}: each panel is "
              "ordered or skipped as a block; the skip probability follows each client's panel profile, and "
              "panels share part of their skip driver in proportion to driver_overlap, so gaps travel "
              "together within panels and partly across them. Seed and realised values above are for the "
              "first run seed.", ""]
    else:
        L += ["", "Clients differ only in which features go missing together; per-feature rates are "
              "matched by exact counts. Seed and realised values above are for the first run seed.", ""]

    L += ["## 3. Computed parameters per client", "",
          f"First run seed, unshrunk clients. Full matrices: `params/<client>.json`. "
          f"Characteristics: {[feats[i] for i in roles['always_observed']]} on frozen design-split decile edges "
          "(in `config.json` -> roles.bin_edges).", ""]
    for c, v in first.items():
        sig = v["sig"]
        L += [f"**{c}** (p_j = {v['n_train'] / sum(x['n_train'] for x in first.values()):.3f})",
              f"- r: {dict(zip(feats, np.round(sig['r'], 3).tolist()))}",
              f"- H top5: {_pairs(sig['H'], feats, 5)}; mean {sig['H'][np.triu_indices(len(feats), 1)].mean():.3f}",
              f"- J top5: {_pairs(sig['J'], feats, 5)}; mean {sig['J'][np.triu_indices(len(feats), 1)].mean():.3f}",
              f"- C top5: {_pairs(sig['C'], feats, 5)}; bottom5: {_pairs(sig['C'], feats, 5, False)}; "
              f"mean {sig['C'][np.triu_indices(len(feats), 1)].mean():.3f}; pairs excluded by constant rule: "
              f"{int(sum(1 for a in range(len(feats)) for b in range(a + 1, len(feats)) if sig['r'][a] in (0, 1) or sig['r'][b] in (0, 1)))}",
              f"- S histograms: " + "; ".join(f"{k}={np.round(hv, 2).tolist()}" for k, hv in v["hist"].items()), ""]

    s0, w0 = s[s.seed == cfg["seeds"][0]], w[w.seed == cfg["seeds"][0]]
    L += ["## 4. Client grouping", "", f"First run seed. Rows = receiver, columns = donor.", ""]
    for sc in ["W_H", "W_C", "s", "S", "Q"]:
        L += [f"**{sc}**", "", s0[s0.score == sc].pivot(index="receiver", columns="donor", values="value")
              .round(3).pipe(_md), ""]
    L += ["**Normalised donor weights per arm**", "",
          w0.pivot_table(index=["receiver", "arm"], columns="donor", values="weight").round(3).pipe(_md), ""]
    for rec, g in s0.groupby("receiver"):
        undefined = [sc for sc in SCORES if g[g.score == sc]["value"].isna().all()]
        tops = {sc: g[g.score == sc].sort_values("value", ascending=False)["donor"].tolist()
                for sc in SCORES if sc not in undefined}
        fb = g[g.Q_source != "formula"]["donor"].unique().tolist()
        agree = "AGREE" if len({tuple(t) for t in tops.values()}) == 1 else "DISAGREE"
        L.append(f"- receiver {rec}: " + "; ".join(f"{sc} favours {t[0]}" for sc, t in tops.items())
                 + f". Donor ordering across defined scores: {agree}."
                 + (f" UNDEFINED: {undefined} - that arm fell back to the size anchor p_j." if undefined else "")
                 + (f" Q fallback fired for donors {fb}." if fb else ""))
    L.append("")
    L += recovery_md({o["metrics"][0]["seed"]: o["params"] for o in outs})
    L += fallback_md(w)

    L += ["## 5. Aggregation and headroom", ""]
    if h.empty:
        head_md = "Headroom: n/a (not computed in this run)."
        h = pd.DataFrame({"receiver": [], "verdict": []})
    else:
        head_md = h.groupby("receiver").agg(
            loss_local=("loss_local", "mean"), loss_pooled=("loss_pooled", "mean"),
            headroom=("headroom", "mean"), rel=("rel_headroom", "mean"),
            pooling_harm=("pooling_harm", "mean"), pooled_worse=("pooling_harm", lambda v: int((v > 0).sum())),
            verdicts=("verdict", lambda v: ", ".join(f"{k} x{int(n)}" for k, n in v.value_counts().items()))
        ).round(4).pipe(_md)
    L += [head_md, "",
          f"- alpha {ac['alpha']}, beta {ac['beta']}, gamma {ac['gamma']} (fedavg: gamma = p_i), "
          f"lambda_pop {ac['lambda_pop']}, adaptation budget {mc['adapt_budget']} steps "
          f"(local-only: {mc['local_steps']} + {mc['adapt_budget']}).",
          "- Headroom references (local, pooled) share one protocol: from theta_0, early-stopped on the "
          "receiver's validation fold, evaluated on its test fold.",
          f"- FedAvg limit check in this run: {'PASSED' if fedavg_limit_ok() else 'FAILED'}.", ""]
    L += compute_md(comp, cfg["clients"]["K"])

    L += ["## 6. Results and ablations", ""]
    if (h.verdict == "NO HEADROOM").any():
        L += ["> Context, not a gate: **NO HEADROOM** (pooled reference) for "
              + ", ".join(sorted(h[h.verdict == 'NO HEADROOM'].receiver.unique()))
              + ". Differences between arms there are expected to be small or unstable under this "
              "configuration; a subset-weighted arm may still differ.", ""]
    if (h.verdict == "POOLING HARMS").any():
        L += ["> **POOLING HARMS** for " + ", ".join(sorted(h[h.verdict == 'POOLING HARMS'].receiver.unique()))
              + ": indiscriminate pooling is worse than local there, beyond sampling noise.", ""]
    for metric in ["rmse", "mae", "auc"]:
        rows = []
        for arm in arms:
            row = {"arm": arm}
            for tp in ["t1", "t2"]:
                g = m[(m.metric == metric) & (m.arm == arm) & (m.timepoint == tp)]
                loc = m[(m.metric == metric) & (m.arm == "local-only") & (m.timepoint == tp)]
                mu, sd = _mean_sd(g)
                row[tp] = f"{mu:.4f} +- {sd:.4f}"
                row[f"delta_{tp}"] = f"{mu - _mean_sd(loc)[0]:+.4f}"
            rows.append(row)
        extra = f" Positive-class rate {m[m.metric == 'auc'].pos_rate.mean():.3f}." if metric == "auc" else ""
        L += [f"**{metric.upper()}** (mean +- std over seeds of the receiver mean).{extra}", "",
              pd.DataFrame(rows).pipe(_md, index=False), ""]
    r2 = m[(m.metric == "rmse") & (m.timepoint == "t2")].pivot_table(index="arm", columns="receiver", values="value")
    delta = r2.sub(r2.loc["local-only"], axis=1)
    per = r2.copy()
    per["mean"], per["worst"] = r2.mean(1), r2.max(1)
    per["frac_harmed"] = (delta > 0).mean(1)
    L += ["**Per-receiver RMSE, timepoint 2** (mean over seeds)", "", per.loc[arms].round(4).pipe(_md), ""]
    L += averaging_harm(m, arms)
    L += contrasts_md(m_all, fold)
    L += selection_md(m_all, cand, fold)
    u = s[s.score == "W_H"].groupby(["receiver", "donor"])[["U_t1", "U_t2"]].mean()
    L += ["**Measured transfer benefit U[i<-j]** (RMSE reduction vs local-only; mean over seeds)", "",
          u.round(4).pipe(_md), ""]
    agree = []
    for sc in SCORES:
        hit1 = hit2 = n = 0
        taus = []
        for (_, _), g in s[s.score == sc].groupby(["seed", "receiver"]):
            if g["value"].isna().any():
                continue
            n += 1
            top = g.loc[g["value"].idxmax(), "donor"]
            hit1 += top == g.loc[g["U_t1"].idxmax(), "donor"]
            hit2 += top == g.loc[g["U_t2"].idxmax(), "donor"]
            taus.append(kendalltau(g["value"], g["U_t1"]).statistic)
        agree.append({"score": sc, "receiver-seeds": n, "top donor = top U (t1)": hit1,
                      "top donor = top U (t2)": hit2, "mean Kendall tau vs U_t1": np.nanmean(taus) if taus else np.nan})
    L += ["**Score-vs-U agreement**", "", pd.DataFrame(agree).round(3).pipe(_md, index=False), "",
          "**Reading**", "", READING, ""]
    return "\n".join(L)


def write_run(exp, cfg, name, df, roles, dropped, outs, root="results"):
    full = {**cfg, "experiment": exp, "dataset": name, "roles": roles}
    out_dir = os.path.join(root, exp, run_id(full))
    write_csvs(out_dir, outs)
    os.makedirs(os.path.join(out_dir, "params"), exist_ok=True)
    for c, v in outs[0]["params"].items():
        with open(os.path.join(out_dir, "params", f"{c}.json"), "w") as f:
            json.dump(_jsonable({**v["sig"], "hist": v["hist"], "n_train": v["n_train"],
                                 "group": v["group"], "profile": v["profile"]}), f)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(_jsonable(full), f, indent=2)
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(build_report(full, name, df, roles, dropped, outs))
    return out_dir
