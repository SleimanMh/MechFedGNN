"""Personalised aggregation kernel (CLAUDE.md §7).

  base_j = alpha q_j + (1 - alpha) p_j          (q_j None -> base_j = p_j)
  w_j    = base_j^beta / sum_{l != i} base_l^beta
  theta  = gamma_i theta_i + (1 - gamma_i) sum_{j != i} w_j theta_j

FedAvg limit: alpha = 0, beta = 1, gamma_i = p_i.
"""
import numpy as np


def donor_weights(i, q, p, alpha, beta=1.0):
    """Normalised donor weights for receiver i; w[i] = 0.

    q: per-client scores (entry i ignored, None where undefined).
    p: client sizes normalised over ALL clients.
    """
    p = np.asarray(p, float)
    base = np.array([p[j] if q[j] is None else alpha * q[j] + (1 - alpha) * p[j]
                     for j in range(len(p))])
    base[i] = 0.0
    base = base ** beta
    total = base.sum()
    if total <= 0:
        raise ValueError(f"receiver {i}: all donor base weights are zero")
    return base / total


def aggregate(thetas, i, w, gamma):
    """Mix parameters. thetas: list of arrays, or list of {name: array} dicts."""
    def mix(values):
        donors = sum(w[j] * values[j] for j in range(len(values)) if j != i)
        return gamma * values[i] + (1 - gamma) * donors

    if isinstance(thetas[0], dict):
        return {k: mix([t[k] for t in thetas]) for k in thetas[0]}
    return mix(thetas)
