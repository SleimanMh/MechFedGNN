"""Headroom diagnostic (CLAUDE.md §8). Evaluation only - never an arm, never a weight source.

headroom = loss_local - loss_pooled, on the receiver's evaluation fold under its
own mask. loss_pooled comes from an oracle trained on the pooled training rows
of all clients (receiver's statistics) - the upper bound for parameter
aggregation.
"""
import numpy as np
import torch

from model import train

NONE_MAX = 0.02
THIN_MAX = 0.10


def verdict(loss_local, loss_pooled, none_max=NONE_MAX, thin_max=THIN_MAX):
    rel = (loss_local - loss_pooled) / loss_local
    if rel <= none_max:
        return "NO HEADROOM"
    if rel <= thin_max:
        return "THIN HEADROOM"
    return "HEADROOM OK"


def pooled_oracle(theta0, d, clients, st, cfg, seed):
    """Train from theta0 on every client's training rows, standardised with the
    receiver's statistics `st`. Step count scales with K so the oracle sees
    about as many epochs as a local model."""
    xs, ys = [], []
    for c in clients:
        tr = c["train"]
        xs.append(st.inputs(c["X"][tr], c["M"][tr]))
        ys.append(st.target(c["y"][tr]))
    mc = cfg["model"]
    steps = len(clients) * (mc["local_steps"] + mc["adapt_budget"]) * mc["pooled_steps_per_client"]
    return train(theta0, d, torch.cat(xs), torch.cat(ys), steps, seed, mc)


def headroom_row(loss_local, loss_pooled, cfg):
    h = cfg.get("headroom", {})
    return {"loss_local": loss_local, "loss_pooled": loss_pooled,
            "headroom": loss_local - loss_pooled,
            "rel_headroom": (loss_local - loss_pooled) / loss_local,
            "verdict": verdict(loss_local, loss_pooled, h.get("none_max", NONE_MAX),
                               h.get("thin_max", THIN_MAX))}


def format_block(rows):
    """Printable headroom block; one line per (seed, receiver) row dict."""
    lines = ["receiver  seed  loss_local  loss_pooled  headroom   rel    verdict"]
    for r in rows:
        lines.append(f"{r['receiver']:>8s}  {r['seed']:>4d}  {r['loss_local']:10.4f}  "
                     f"{r['loss_pooled']:11.4f}  {r['headroom']:+8.4f}  {r['rel_headroom']:+.3f}  "
                     f"{r['verdict']}")
    worst = [r for r in rows if r["verdict"] == "NO HEADROOM"]
    if worst:
        lines.insert(0, f"!!! NO HEADROOM for {len(worst)} receiver-seed(s): results there are "
                        "noise around the receiver's own optimum")
    return "\n".join(lines)


def summarise(rows):
    """Count of verdicts, for pilot tables."""
    v = [r["verdict"] for r in rows]
    return {k: v.count(k) for k in ["HEADROOM OK", "THIN HEADROOM", "NO HEADROOM"]} | {
        "median_rel": float(np.median([r["rel_headroom"] for r in rows]))}
