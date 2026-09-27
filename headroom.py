"""Headroom diagnostic (CLAUDE.md §8). Evaluation only - never an arm, never a weight source.

headroom = loss_local - loss_pooled on the receiver's TEST fold under its own
mask. Both reference models use the SAME stopping rule - from theta_0,
early-stopped on the receiver's validation fold, receiver's standardisation.
Identical stopping criteria are not equal compute: steps and examples processed
are returned and reported separately.
  local  : the receiver's own training rows
  pooled : the training rows of all clients - one training procedure, NOT an
           upper bound for personalised aggregation (it cannot weight a subset
           of donors). Headroom is a reference reported beside results, never a
           gate on any experiment or dataset.

Downside is reported too: pooling_harm = loss_pooled - loss_local where
positive. POOLING HARMS is declared only beyond sampling noise, judged on the
paired per-row squared errors of the two references on the same test rows:
  d_r = e_pooled,r^2 - e_local,r^2,   harm if mean(d) > harm_se * sd(d)/sqrt(n).
"""
import numpy as np
import torch

from model import predict, train_early_stopping

NONE_MAX = 0.02
THIN_MAX = 0.10
HARM_SE = 2.0
VERDICTS = ["HEADROOM OK", "THIN HEADROOM", "NO HEADROOM", "POOLING HARMS"]


def verdict(loss_local, loss_pooled, harm_z=0.0, none_max=NONE_MAX, thin_max=THIN_MAX, harm_se=HARM_SE):
    """harm_z: paired z-statistic of (pooled - local) squared error; > harm_se = real harm."""
    if loss_pooled > loss_local and harm_z > harm_se:
        return "POOLING HARMS"
    rel = (loss_local - loss_pooled) / loss_local
    if rel <= none_max:
        return "NO HEADROOM"
    if rel <= thin_max:
        return "THIN HEADROOM"
    return "HEADROOM OK"


def _fold(c, st, fold):
    rows = c[fold]
    return st.inputs(c["X"][rows], c["M"][rows]), c["y"][rows]


def reference_losses(theta0, d, rec, clients, st, cfg, seed):
    """Test errors of the early-stopped local and pooled references, with their
    best and halting steps."""
    mc, es = cfg["model"], cfg["headroom"]["early_stopping"]
    xval, yval = _fold(rec, st, "val")
    xte, yte = _fold(rec, st, "test")
    yval_t = st.target(yval)
    pooled = [_fold(c, st, "train") for c in clients]
    sets = {"local": _fold(rec, st, "train"),
            "pooled": (torch.cat([x for x, _ in pooled]), np.concatenate([y for _, y in pooled]))}
    out = {}
    for name, (x, y) in sets.items():
        params, best, halt = train_early_stopping(theta0, d, x, st.target(y), xval, yval_t, seed, mc, es)
        out[f"{name}_err"] = predict(params, d, xte, st, tuple(mc["hidden"])) - yte
        out[f"{name}_best"], out[f"{name}_halt"] = best, halt
    return out


def headroom_row(ref, cfg):
    h = cfg.get("headroom", {})
    el, ep = ref["local_err"], ref["pooled_err"]
    loss_local, loss_pooled = float(np.sqrt((el ** 2).mean())), float(np.sqrt((ep ** 2).mean()))
    dsq = ep ** 2 - el ** 2
    se = dsq.std(ddof=1) / np.sqrt(len(dsq)) if len(dsq) > 1 else float("inf")
    harm_z = float(dsq.mean() / se) if se > 0 else 0.0
    return {"loss_local": loss_local, "loss_pooled": loss_pooled,
            "headroom": loss_local - loss_pooled,
            "rel_headroom": (loss_local - loss_pooled) / loss_local,
            "pooling_harm": max(loss_pooled - loss_local, 0.0), "harm_z": harm_z,
            "local_best": ref["local_best"], "local_halt": ref["local_halt"],
            "pooled_best": ref["pooled_best"], "pooled_halt": ref["pooled_halt"],
            "verdict": verdict(loss_local, loss_pooled, harm_z, h.get("none_max", NONE_MAX),
                               h.get("thin_max", THIN_MAX), h.get("harm_se", HARM_SE))}


def format_block(rows):
    """Printable headroom block; one line per (seed, receiver) row dict."""
    lines = ["receiver  seed  loss_local  loss_pooled  headroom   rel    harm   harm_z  "
             "best/halt local  best/halt pooled  verdict"]
    for r in rows:
        lines.append(f"{r['receiver']:>8s}  {r['seed']:>4d}  {r['loss_local']:10.4f}  "
                     f"{r['loss_pooled']:11.4f}  {r['headroom']:+8.4f}  {r['rel_headroom']:+.3f}  "
                     f"{r['pooling_harm']:.4f}  {r['harm_z']:+6.2f}  "
                     f"{r['local_best']:>5d}/{r['local_halt']:<5d}      {r['pooled_best']:>5d}/{r['pooled_halt']:<5d}"
                     f"       {r['verdict']}")
    harmed = sum(r["pooling_harm"] > 0 for r in rows)
    real = sum(r["verdict"] == "POOLING HARMS" for r in rows)
    lines.append(f"pooling worse than local in {harmed}/{len(rows)} receiver-seeds; "
                 f"beyond noise (POOLING HARMS) in {real}/{len(rows)}")
    none = sum(r["verdict"] == "NO HEADROOM" for r in rows)
    if none:
        lines.insert(0, f"context: NO HEADROOM (pooled reference) for {none} receiver-seed(s); "
                        "arm differences there are expected to be small or unstable")
    return "\n".join(lines)
