"""Parameter geometry logged during a run (CLAUDE.md §10, hypotheses A / B).

All distances are L2 over the flattened parameter vector of the mask-aware MLP.
Mixtures are measured BEFORE adaptation (timepoint 1), which is where the
weighting acts.
"""
import numpy as np


def vec(params):
    return np.concatenate([np.ravel(params[k]) for k in sorted(params)])


def client_rows(seed, thetas, theta0, ids):
    """Pairwise distances between the clients' local parameters, and each one's
    distance from theta_0 (the scale the pairwise distances are read against)."""
    v = [vec(t) for t in thetas]
    v0 = vec(theta0)
    rows = [{"seed": seed, "receiver": "-", "kind": "client-client", "a": ids[a], "b": ids[b],
             "dist": float(np.linalg.norm(v[a] - v[b]))}
            for a in range(len(v)) for b in range(a + 1, len(v))]
    rows += [{"seed": seed, "receiver": "-", "kind": "client-theta0", "a": ids[a], "b": "theta0",
              "dist": float(np.linalg.norm(v[a] - v0))} for a in range(len(v))]
    return rows


def mixture_rows(seed, receiver, thetas, i, mixtures, singles):
    """mixtures: {arm: params} (receiver-weighted mixtures incl. uniform-donor);
    singles: {donor_id: params} single-donor mixes. Distances to the unweighted
    centroid of all local models, to the uniform-donor mixture, and to the
    receiver's own local parameters."""
    v = [vec(t) for t in thetas]
    centroid = np.mean(v, axis=0)
    own = v[i]
    uni = vec(mixtures["uniform-donor"])
    rows = []
    for kind, items in [("arm", mixtures), ("single", singles)]:
        for name, params in items.items():
            x = vec(params)
            rows.append({"seed": seed, "receiver": receiver, "kind": kind, "a": name, "b": "-",
                         "to_centroid": float(np.linalg.norm(x - centroid)),
                         "to_uniform_mix": float(np.linalg.norm(x - uni)),
                         "to_receiver_local": float(np.linalg.norm(x - own))})
    return rows


def _rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def function_rows(seed, receiver, client_preds, preds, main_b, local_rmse):
    """Functional comparison on the receiver's test inputs, shared coordinates
    (§16): RMS prediction difference between every pair of client local models,
    and between each arm / single-donor mix and the uniform-donor mixture (and
    local-only) at t1 and t2. `rel` divides by the receiver's local-only test RMSE."""
    rows = []
    ids = sorted(client_preds)
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            v = _rms(client_preds[ids[x]], client_preds[ids[y]])
            rows.append({"seed": seed, "receiver": receiver, "kind": "client-client", "a": ids[x],
                         "b": ids[y], "timepoint": "-", "rms": v, "rel": v / local_rmse})
    uni, loc = preds["adapt:uniform-donor"], preds["adapt:local-only"]
    for comp, per_step in preds.items():
        kind, name = comp.split(":", 1)
        for step, tp in [(0, "t1"), (main_b, "t2")]:
            refs = [("uniform-donor", uni)] + ([("local-only", loc)] if kind == "candidate" else [])
            for ref_name, ref in refs:
                if comp == f"adapt:{ref_name}":
                    continue
                v = _rms(per_step[step], ref[step])
                rows.append({"seed": seed, "receiver": receiver, "kind": "arm" if kind == "adapt" else "single",
                             "a": name, "b": ref_name, "timepoint": tp, "rms": v, "rel": v / local_rmse})
    return rows
