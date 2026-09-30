"""Parameter aggregation ONLY.

Combines compatible model states with weights it is GIVEN. It does not score,
does not normalise and does not decide participation.

    theta_new = gamma * theta_receiver + (1 - gamma) * sum_{j != i} w_j theta_j
"""
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from kernel import aggregate


def schema_of(state: Mapping[str, np.ndarray]) -> dict:
    """Tensor names, shapes and dtypes. Two states are compatible only if their
    schemas match exactly; used to reject incompatible updates before they
    enter aggregation."""
    return {k: [list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in sorted(state.items())}


class IncompatibleState(ValueError):
    pass


def check_compatible(states: Sequence[Mapping[str, np.ndarray]]) -> dict:
    ref = schema_of(states[0])
    for i, s in enumerate(states[1:], start=1):
        if schema_of(s) != ref:
            raise IncompatibleState(f"state {i} schema does not match state 0")
    return ref


@dataclass
class WeightedAverage:
    """The aggregator used by every arm; arms differ only in their weights."""
    strict: bool = True

    def combine(self, states: Sequence[Mapping[str, np.ndarray]], receiver: int,
                weights: np.ndarray, gamma: float) -> Any:
        if self.strict:
            check_compatible(states)
            w = np.asarray(weights, float)
            if w[receiver] != 0.0:
                raise ValueError("donor weights must be zero at the receiver index")
            if not np.isfinite(w).all():
                raise ValueError("non-finite donor weight")
            if abs(w.sum() - 1.0) > 1e-9:
                raise ValueError(f"donor weights must sum to 1, got {w.sum()!r}")
        return aggregate(list(states), receiver, weights, gamma)
