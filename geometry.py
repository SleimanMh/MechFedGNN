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
