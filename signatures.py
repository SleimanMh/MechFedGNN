"""Per-client mask summaries and population histograms (CLAUDE.md §5). Pure NumPy.

Mask convention: M = 1 observed, M = 0 missing; absence A = 1 - M.

H, C and J are different things and must never be conflated:
  H[f,g] how often a joint gap happens        P(A_f = 1, A_g = 1)
  C[f,g] how associated two gaps are          phi between A_f and A_g
  J[f,g] how often a sender sees the pair     P(M_f = 1, M_g = 1)
"""
import numpy as np


def _absence(M):
    return 1.0 - np.asarray(M, dtype=float)


def rates(M):
    """r[f] = P(A_f = 1)."""
    return _absence(M).mean(0)


def joint_absence(M):
    """H[f,g] = P(A_f = 1, A_g = 1). Diagonal equals r."""
    A = _absence(M)
    return A.T @ A / len(A)


def joint_observation(M):
    """J[f,g] = P(M_f = 1, M_g = 1) = 1 - r_f - r_g + H[f,g]."""
    O = np.asarray(M, dtype=float)
    return O.T @ O / len(O)


def phi(M):
    """C[f,g] = (H - r_f r_g) / sqrt(r_f(1-r_f) r_g(1-r_g)), in [-1, 1], diagonal 0.

    A feature that is always observed or always missing has undefined phi; its
    row and column are set to 0 (its rate is still available from `rates`).
    """
    r, H = rates(M), joint_absence(M)
    v = r * (1 - r)
    den = np.sqrt(np.outer(v, v))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(den > 0, (H - np.outer(r, r)) / den, 0.0)
    np.fill_diagonal(C, 0.0)
    return np.clip(C, -1.0, 1.0)


def chi2(M):
    """Pairwise chi-square statistic n * C^2 (phi^2 = chi2 / n)."""
    return len(M) * phi(M) ** 2


def signature(M):
    """All mask summaries of one client's training mask."""
    M = np.asarray(M)
    C = phi(M)
    return {"n": len(M), "r": rates(M), "H": joint_absence(M), "J": joint_observation(M),
            "C": C, "chi2": len(M) * C**2}


# ---------------------------------------------------------------- population

def histograms(X, names, edges):
    """Normalised histogram per characteristic on shared, frozen bin edges.

    X columns align with `names`; `edges[name]` are the frozen bin edges.
    Values outside the edges are clipped into the end bins.
    """
    out = {}
    for j, name in enumerate(names):
        e = np.asarray(edges[name], float)
        if len(e) < 2:
            continue
        counts, _ = np.histogram(np.clip(X[:, j], e[0], e[-1]), bins=e)
        out[name] = counts / counts.sum()
    return out


def population_similarity(h_i, h_j):
    """S(i,j) = mean over shared characteristics of sum_b min(h_i[b], h_j[b]).

    A characteristic present in only one client is dropped from the mean, never
    scored 0. No shared characteristic -> None.
    """
    shared = sorted(set(h_i) & set(h_j))
    if not shared:
        return None
    return float(np.mean([np.minimum(h_i[u], h_j[u]).sum() for u in shared]))
