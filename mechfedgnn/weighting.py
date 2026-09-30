"""Weighting policies - weight normalization ONLY.

Takes scores and sample sizes, returns (donor weights, self-weight gamma,
fallback reason). Owns sample-size blending (alpha), sharpening (beta),
normalization and the declared fallback. It never computes a score and never
touches model parameters.

Three separate quantities, never merged:
  * ``p``     sample-size weights, normalised over ALL clients;
  * ``w``     donor weights, normalised over the donors (w[receiver] = 0);
  * ``gamma`` the receiver's self-weight.

Declared fallback (CLAUDE.md 7.1): if a score is undefined for ANY donor, or
all donor scores are equal within ``tol``, the score provides no basis for
distinguishing donors and the arm falls back to SAMPLE-SIZE weighting, keeping
its own gamma. The reason is recorded, never silently applied.

FedAvg limit: alpha = 0, beta = 1, gamma_i = p_i reproduces
``theta = sum_j p_j theta_j`` (unit-tested to 1e-8). No claim is made that any
particular MCAR mask automatically recovers FedAvg.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kernel import donor_weights
from mechfedgnn.registry import WEIGHTING

NONDISCRIM_TOL = 1e-9


@dataclass
class LocalOnly:
    name: str = "local-only"

    def weights(self, scores, sizes, receiver):
        return None, 1.0, ""


@dataclass
class FedAvg:
    """alpha = 0 (sizes only), beta = 1, gamma = p_receiver."""
    name: str = "fedavg"

    def weights(self, scores, sizes, receiver):
        p = np.asarray(sizes, float)
        return donor_weights(receiver, [None] * len(p), p, alpha=0.0, beta=1.0), float(p[receiver]), ""


@dataclass
class UniformDonor:
    """Equal donor weights, with the SAME self-weight the score arms use, so a
    score arm is compared against equal weighting and not against a different
    self-weight."""
    gamma: float = 0.5
    name: str = "uniform-donor"

    def weights(self, scores, sizes, receiver):
        k = len(sizes)
        w = np.full(k, 1.0 / (k - 1))
        w[receiver] = 0.0
        return w, float(self.gamma), ""


@dataclass
class ScoreWeighting:
    """base_j = alpha q_j + (1 - alpha) p_j ; w_j proportional to base_j^beta."""
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    tol: float = NONDISCRIM_TOL
    name: str = "score"

    def weights(self, scores: Mapping[int, float | None], sizes, receiver):
        p = np.asarray(sizes, float)
        k = len(p)
        q = [None if j == receiver else scores.get(j) for j in range(k)]
        donors = [q[j] for j in range(k) if j != receiver]
        if any(v is None for v in donors):
            reason = "undefined"
        elif max(donors) - min(donors) <= self.tol:
            reason = "non-discriminating"
        else:
            return (donor_weights(receiver, q, p, alpha=self.alpha, beta=self.beta),
                    float(self.gamma), "")
        # declared fallback: sample-size weighting, arm's own gamma, reason recorded
        return (donor_weights(receiver, [None] * k, p, alpha=0.0, beta=1.0),
                float(self.gamma), reason)


WEIGHTING.register("local-only", lambda **kw: LocalOnly())
WEIGHTING.register("fedavg", lambda **kw: FedAvg())
WEIGHTING.register("uniform-donor", lambda gamma=0.5, **kw: UniformDonor(gamma=gamma))
WEIGHTING.register("score", lambda alpha=1.0, beta=1.0, gamma=0.5, tol=NONDISCRIM_TOL, **kw:
                   ScoreWeighting(alpha=alpha, beta=beta, gamma=gamma, tol=tol))


def policy_for(arm: str, aggregation_cfg: Mapping[str, float]):
    """The policy an arm uses. Scoreless arms have their own policy; every score
    arm shares ScoreWeighting, so scoring and weighting stay independent."""
    ac = aggregation_cfg
    if arm in ("local-only", "fedavg"):
        return WEIGHTING.create(arm)
    if arm == "uniform-donor":
        return WEIGHTING.create("uniform-donor", gamma=ac["gamma"])
    return WEIGHTING.create("score", alpha=ac["alpha"], beta=ac["beta"], gamma=ac["gamma"])
