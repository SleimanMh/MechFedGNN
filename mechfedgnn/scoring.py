"""Scoring strategies - score calculation ONLY.

Each strategy maps one (receiver summary, donor summary) pair to a single score
or ``None``. No normalization, no sample-size blending, no aggregation.

Definitions are delegated verbatim to the research modules, so the registered
methods keep exactly the behaviour the completed experiments used.

Note the two MARGINAL methods are DIFFERENT and both are kept:
  * ``marginal-rate``      0.5 (1 + cos(r_i, r_j))     SYMMETRIC similarity
  * ``coverage-marginal``  sum r_i (1 - r_j) / sum r_i  DIRECTED coverage
"""
from dataclasses import dataclass
from typing import Any, Mapping

from mechfedgnn.registry import SCORES
from scores import (combined_q, missingness_similarity, rate_similarity, w_c, w_h,
                    w_marginal)
from signatures import population_similarity

Summary = Mapping[str, Any]


@dataclass
class _Fn:
    """Wraps a research function as a ScoringStrategy."""
    name: str
    fn: Any

    def score(self, receiver: Summary, donor: Summary) -> float | None:
        return self.fn(receiver, donor)


def _w_h(i, j):
    return w_h(i["sig"]["H"], j["sig"]["J"])[0]


def _w_c(i, j):
    return w_c(i["sig"]["C"], j["sig"]["J"])[0]


def _w_marginal(i, j):
    return w_marginal(i["sig"]["r"], j["sig"]["r"])[0]


def _missingness_similarity(i, j):
    return missingness_similarity(i["sig"]["C"], j["sig"]["C"])[0]


def _rate_similarity(i, j):
    return rate_similarity(i["sig"]["r"], j["sig"]["r"])[0]


def _population_similarity(i, j):
    return population_similarity(i["hist"], j["hist"])


def _combined(inner, lam_pop):
    """Q = (1 - lam) * inner + lam * S, with the declared fallback hierarchy
    (both -> formula; W undefined -> S; S undefined -> W; neither -> None)."""
    def fn(i, j):
        return combined_q(inner(i, j), _population_similarity(i, j), lam_pop)[0]
    return fn


SCORES.register("coverage-W_H", lambda **kw: _Fn("coverage-W_H", _w_h))
SCORES.register("coverage-W_C", lambda **kw: _Fn("coverage-W_C", _w_c))
SCORES.register("coverage-marginal", lambda **kw: _Fn("coverage-marginal", _w_marginal))
SCORES.register("missingness-similarity",
                lambda **kw: _Fn("missingness-similarity", _missingness_similarity))
SCORES.register("marginal-rate", lambda **kw: _Fn("marginal-rate", _rate_similarity))
SCORES.register("population-S", lambda **kw: _Fn("population-S", _population_similarity))
SCORES.register("combined-Q",
                lambda lambda_pop=0.5, **kw: _Fn("combined-Q", _combined(_w_h, lambda_pop)))
SCORES.register("combined-Q-marginal",
                lambda lambda_pop=0.5, **kw: _Fn("combined-Q-marginal",
                                                 _combined(_w_marginal, lambda_pop)))

# Arms that use no score at all (their weighting policy needs no ScoringStrategy).
SCORELESS_ARMS = ("local-only", "fedavg", "uniform-donor")
