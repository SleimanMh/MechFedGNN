"""Client construction (CLAUDE.md §4).

Stage 0/1 scope: latent missingness-group profiles and homogeneous row
assignment from the non-design pool. Stratified assignment, characteristics
and receiver shrinking land in Stage 2.
"""
import numpy as np

from data.design import design_split

P_RARE = 0.2
P_USUAL = 0.9
PROFILE_NOISE = 0.05


def group_profiles(n_panels, p_rare=P_RARE, p_usual=P_USUAL):
    """(G=2, n_panels) ordering probabilities.

    Group A rarely orders the first half of the ordered panel list, group B the
    second half. With an odd count the middle panel stays usually-ordered for
    both groups - a within-design control that no score should favour.
    """
    half = n_panels // 2
    a = np.full(n_panels, p_usual)
    b = np.full(n_panels, p_usual)
    a[:half] = p_rare
    b[n_panels - half:] = p_rare
    return np.stack([a, b])


def client_profiles(n_panels, rng, K=6, G=2, noise=PROFILE_NOISE):
    """Per-client profiles = group profile + U(-noise, noise), clipped to [0.05, 0.95].

    Returns (profiles (K, n_panels), groups (K,)). Group labels are ground truth
    for evaluation only - never an input to any score.
    """
    groups = np.repeat(np.arange(G), K // G)
    base = group_profiles(n_panels)[groups]
    prof = np.clip(base + rng.uniform(-noise, noise, base.shape), 0.05, 0.95)
    return prof, groups


def client_pool(n):
    """Rows eligible for clients: everything outside the design split."""
    return design_split(n)[1]


def assign_rows_homogeneous(n, K, rng):
    """Shuffle the pool and split it into K disjoint, near-equal parts."""
    pool = rng.permutation(client_pool(n))
    return [np.sort(part) for part in np.array_split(pool, K)]
