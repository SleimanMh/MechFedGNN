"""Pairwise collaboration scores (CLAUDE.md §6). All in [0, 1].

W_H and W_C are directed (receiver i <- sender j); s, rate and S are symmetric.
Every score returns (value, defined); an undefined score is (None, False).
"""
import numpy as np


def _upper(A):
    return A[np.triu_indices(A.shape[0], 1)]


def _weighted_coverage(weights, J_j):
    w = _upper(weights)
    den = w.sum()
    if den <= 0:
        return None, False
    return float((w * _upper(J_j)).sum() / den), True


def w_h(H_i, J_j):
    """W_H(i<-j) = sum_{f<g} H_i[f,g] J_j[f,g] / sum_{f<g} H_i[f,g].

    Means: the sender observes together the pairs the receiver frequently lacks
    together. It does NOT measure feature importance and does NOT establish that
    the sender's model holds transferable knowledge - that is the hypothesis
    under test.
    """
    return _weighted_coverage(H_i, J_j)


def w_c(C_i, J_j):
    """W_C(i<-j): as W_H, weighted by positive gap association max(C_i, 0).

    Same caveat as W_H: coverage of associated gaps, not proven usefulness.
    """
    return _weighted_coverage(np.maximum(C_i, 0.0), J_j)


def _cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None, False
    return float(np.clip(0.5 * (1 + a @ b / (na * nb)), 0.0, 1.0)), True


def missingness_similarity(C_i, C_j):
    """s(i,j) = 0.5 (1 + cosine(upper C_i, upper C_j))."""
    return _cosine_similarity(_upper(C_i), _upper(C_j))


def rate_similarity(r_i, r_j):
    """rate(i,j) = 0.5 (1 + cosine(r_i, r_j))."""
    return _cosine_similarity(np.asarray(r_i, float), np.asarray(r_j, float))


def combined_q(W, S, lam_pop):
    """Q(i<-j) = (1 - lam_pop) W + lam_pop S, with the fallback hierarchy.

    Returns (value, source), source in {"formula", "S", "W", "none"} so the
    report can say which fallback fired. "none" -> kernel uses the size anchor.
    """
    if W is not None and S is not None:
        return (1 - lam_pop) * W + lam_pop * S, "formula"
    if W is None and S is not None:
        return S, "S"
    if W is not None:
        return W, "W"
    return None, "none"
