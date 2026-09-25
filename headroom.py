"""Headroom diagnostic (CLAUDE.md §8). Evaluation only - never an arm, never a weight source.

headroom = loss_local - loss_pooled on the receiver's TEST fold under its own
mask. Both reference models are trained under the SAME protocol - from theta_0,
early-stopped on the receiver's validation fold, receiver's standardisation -
so the gap measures information in the pooled rows, not step count:
  local  : the receiver's own training rows
  pooled : the training rows of all clients (upper bound for aggregation)
"""
import numpy as np
import torch

from model import metrics, predict, train_early_stopping

NONE_MAX = 0.02
THIN_MAX = 0.10


def verdict(loss_local, loss_pooled, none_max=NONE_MAX, thin_max=THIN_MAX):
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
    """Test RMSE of the early-stopped local and pooled references, and their stop steps."""
    mc, es = cfg["model"], cfg["headroom"]["early_stopping"]
    xval, yval = _fold(rec, st, "val")
    xte, yte = _fold(rec, st, "test")
    yval_t = st.target(yval)
    pooled = [_fold(c, st, "train") for c in clients]
    sets = {"local": _fold(rec, st, "train"),
            "pooled": (torch.cat([x for x, _ in pooled]), np.concatenate([y for _, y in pooled]))}
    out = {}
    for name, (x, y) in sets.items():
        params, step = train_early_stopping(theta0, d, x, st.target(y), xval, yval_t, seed, mc, es)
        out[name] = metrics(yte, predict(params, d, xte, st, tuple(mc["hidden"])), st.y_median)["rmse"]
        out[f"{name}_stop"] = step
    return out


def headroom_row(ref, cfg):
    h = cfg.get("headroom", {})
    loss_local, loss_pooled = ref["local"], ref["pooled"]
    return {"loss_local": loss_local, "loss_pooled": loss_pooled,
            "headroom": loss_local - loss_pooled,
            "rel_headroom": (loss_local - loss_pooled) / loss_local,
            "local_stop": ref["local_stop"], "pooled_stop": ref["pooled_stop"],
            "verdict": verdict(loss_local, loss_pooled, h.get("none_max", NONE_MAX),
                               h.get("thin_max", THIN_MAX))}


def format_block(rows):
    """Printable headroom block; one line per (seed, receiver) row dict."""
    lines = ["receiver  seed  loss_local  loss_pooled  headroom   rel    stop(local/pooled)  verdict"]
    for r in rows:
        lines.append(f"{r['receiver']:>8s}  {r['seed']:>4d}  {r['loss_local']:10.4f}  "
                     f"{r['loss_pooled']:11.4f}  {r['headroom']:+8.4f}  {r['rel_headroom']:+.3f}  "
                     f"{r['local_stop']:>7d}/{r['pooled_stop']:<7d}    {r['verdict']}")
    worst = [r for r in rows if r["verdict"] == "NO HEADROOM"]
    if worst:
        lines.insert(0, f"!!! NO HEADROOM for {len(worst)} receiver-seed(s): results there are "
                        "noise around the receiver's own optimum")
    return "\n".join(lines)
