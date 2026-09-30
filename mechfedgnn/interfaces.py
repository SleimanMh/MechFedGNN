"""Component interfaces for the MechFedGNN federated research framework.

Milestone A: these describe the boundaries. Implementations delegate to the
existing research code, so behaviour is unchanged by construction and the
golden-reference tests prove it.

Conventions that every implementation must honour (unchanged from the research
code, CLAUDE.md sections 5-7):

  * ``M = 1`` means OBSERVED; absence is ``A = 1 - M``.
  * ``H[f,g] = P(A_f = 1, A_g = 1)``  joint ABSENCE.
  * ``C[f,g]``                        binary association (phi) between absences;
                                      a feature that is always observed or always
                                      missing has undefined phi -> its row/column
                                      is 0, and its rate is kept separately in r.
  * ``J[f,g] = P(M_f = 1, M_g = 1)``  joint OBSERVATION.
  * An undefined statistic yields an undefined score (``None``), which the
    weighting policy turns into the declared sample-size fallback.
  * Self-weight ``gamma``, sample-size weights ``p`` and donor weights ``w`` are
    three separate quantities and are never merged.

Score calculation, weight normalization and parameter aggregation are three
DIFFERENT operations and live behind three different interfaces.
"""
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

# A model state is a mapping name -> ndarray, exactly what the learner exports.
ModelState = Mapping[str, np.ndarray]
# A client summary is whatever the signature provider produces for one client.
Summary = Mapping[str, Any]


class DataProvider(Protocol):
    """Loads data and exposes each client's permitted records - nothing else."""

    def client_ids(self) -> Sequence[str]: ...

    def records(self, client_id: str) -> Any:
        """The records this client is permitted to see. A server coordinator
        must never call this for a client it does not own."""


class SplitPolicy(Protocol):
    """Duplicate-group-preserving splits, data roles and compatible transforms."""

    def folds(self, client_id: str) -> Mapping[str, np.ndarray]:
        """train / val / test index arrays, local to the client's records."""

    def transform(self, client_id: str) -> Any:
        """The fitted feature/target transform. Where it was fitted is recorded
        in provenance (shared design-split fit, or per-client fit)."""


class Injector(Protocol):
    """OPTIONAL experimental masks. A client whose data is already incomplete
    must be able to train with no injector at all."""

    def mask(self, client_id: str, n_rows: int, folds: Mapping[str, np.ndarray]) -> np.ndarray | None:
        """``M`` with 1 = observed, or None for 'use the data as it arrives'."""

    def validate(self) -> Mapping[str, Any]:
        """Design checks for THIS experiment's construction. Checks are
        per-experiment: a matched-rate requirement must not be applied to an
        experiment that deliberately gives clients different rates."""


class LocalLearner(Protocol):
    """Trains, adapts, predicts and exports parameters. The active learner is
    the mask-aware MLP with prediction loss only; this is the extension point
    for other learners (no GNN / reconstruction head / attention here)."""

    def initial_state(self, n_features: int, seed: int) -> ModelState: ...

    def train(self, state: ModelState, data: Any, steps: int, seed: int) -> ModelState: ...

    def predict(self, state: ModelState, data: Any) -> np.ndarray: ...

    def schema(self, state: ModelState) -> Mapping[str, Any]:
        """Tensor names, shapes and dtypes - used to reject incompatible updates."""


class SignatureProvider(Protocol):
    """Marginal rates, H, C, J and the permitted population summaries."""

    def summarize(self, client: Any, roles: Mapping[str, Any],
                  exclude: Sequence[int] = ()) -> Summary: ...


class ScoringStrategy(Protocol):
    """Turns a receiver/donor summary pair into ONE score. Does not normalise,
    does not blend with sample size, does not aggregate."""

    name: str

    def score(self, receiver: Summary, donor: Summary) -> float | None:
        """``None`` means undefined for this pair (documented handling)."""


class WeightingPolicy(Protocol):
    """Turns scores + sample sizes into donor weights and a self-weight.
    Owns sample-size blending, sharpening, normalization and the fallback."""

    name: str

    def weights(self, scores: Mapping[int, float | None], sizes: np.ndarray,
                receiver: int) -> tuple[np.ndarray | None, float, str]:
        """(donor weights or None for local-only, gamma, fallback reason)."""


class Aggregator(Protocol):
    """Combines COMPATIBLE model states using explicit weights. Performs no
    scoring and no normalization - it is handed the weights it must use."""

    def combine(self, states: Sequence[ModelState], receiver: int,
                weights: np.ndarray, gamma: float) -> ModelState: ...


class Evaluator(Protocol):
    """Local prediction metrics and controlled transfer diagnostics."""

    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                train_median: float) -> Mapping[str, float]: ...


class ArtifactStore(Protocol):
    """Saves configuration, provenance, metrics, weights and checkpoints.
    Never silently overwrites a completed run."""

    def write_run(self, run_id: str, payload: Mapping[str, Any]) -> str: ...
