"""Local learner: the mask-aware MLP with prediction loss only.

Delegates to ``model.py`` so training behaviour is unchanged (same seeds,
batches, initialisation and optimizer). This is the extension point for other
learners; no GNN, reconstruction head or attention is implemented.
"""
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mechfedgnn.registry import LEARNERS
from model import build, get_params, init_params, metrics, predict, train


@dataclass
class MaskAwareMLP:
    """Input is concat([x_zero_filled, M]); loss is prediction loss only."""
    hidden: Sequence[int] = (64, 32)
    lr: float = 1e-3
    batch: int = 64
    name: str = "mask-aware-mlp"

    def _cfg(self) -> dict:
        return {"hidden": tuple(self.hidden), "lr": self.lr, "batch": self.batch}

    def initial_state(self, n_features: int, seed: int) -> dict:
        return init_params(n_features, seed, tuple(self.hidden))

    def train(self, state, n_features, xin, yt, steps, seed, checkpoints=(), on_checkpoint=None):
        return train(state, n_features, xin, yt, steps, seed, self._cfg(),
                     checkpoints, on_checkpoint)

    def predict(self, state, n_features, xin, transform) -> np.ndarray:
        return predict(state, n_features, xin, transform, tuple(self.hidden))

    def metrics(self, y_true, y_pred, train_median) -> Mapping[str, float]:
        return metrics(y_true, y_pred, train_median)

    def schema(self, state: Mapping[str, np.ndarray]) -> dict:
        return {k: [list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in sorted(state.items())}

    def export(self, model) -> dict:
        return get_params(model)

    def load(self, state, n_features):
        return build(state, n_features, tuple(self.hidden))


LEARNERS.register("mask-aware-mlp",
                  lambda hidden=(64, 32), lr=1e-3, batch=64, **kw:
                  MaskAwareMLP(hidden=hidden, lr=lr, batch=batch))
