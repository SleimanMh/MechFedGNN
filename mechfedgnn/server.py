"""Server coordinator: clients, rounds, update validation and aggregation.

What the server may receive (and nothing else):
  * model updates;
  * declared sample counts;
  * aggregate signatures required by the selected method;
  * explicitly enabled aggregate metrics.

The server NEVER loads client features, labels or per-row masks. It holds no
DataProvider.

Personalised federation: each client is assigned its OWN parent model, so an
update is validated against THAT client's assignment - not against one global
version.

Declared policies (never silent):
  * participation - every configured client must complete the round;
  * timeout       - on deadline, FAIL or PAUSE explicitly; participants are not
                    quietly changed and weights are not renormalised;
  * duplicate     - an update_id already accepted this round is rejected;
  * late/stale    - an update for a closed or mismatched round is rejected.
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mechfedgnn.aggregation import WeightedAverage, schema_of
from mechfedgnn.protocol import Envelope, PROTOCOL_VERSION, ProtocolError, schema_id, state_version


class Reject(str, Enum):
    PROTOCOL = "protocol_version_mismatch"
    EXPERIMENT = "unknown_experiment"
    IDENTITY = "identity_mismatch"
    UNKNOWN_CLIENT = "unknown_client"
    ROUND = "stale_or_late_round"
    PARENT = "parent_version_mismatch"
    SCHEMA = "schema_mismatch"
    DUPLICATE = "duplicate_update"
    MALFORMED = "malformed_payload"


class RoundTimeout(RuntimeError):
    pass


class UpdateRejected(ValueError):
    def __init__(self, reason: Reject, detail: str = ""):
        super().__init__(f"{reason.value}{': ' + detail if detail else ''}")
        self.reason = reason


@dataclass
class RoundState:
    round_id: int
    assigned: dict[str, str] = field(default_factory=dict)     # client -> parent version
    updates: dict[str, dict] = field(default_factory=dict)     # client -> state
    counts: dict[str, int] = field(default_factory=dict)       # client -> declared n
    signatures: dict[str, dict] = field(default_factory=dict)  # client -> AGGREGATE summary
    accepted_ids: set[str] = field(default_factory=set)
    closed: bool = False
    opened_at: float = field(default_factory=time.monotonic)


@dataclass
class ServerCoordinator:
    experiment_id: str
    clients: Sequence[str]
    schema: str
    round_timeout_s: float = 60.0
    on_timeout: str = "fail"                 # "fail" | "pause"
    aggregator: Any = field(default_factory=WeightedAverage)
    models: dict[str, dict] = field(default_factory=dict)      # client -> its parent model
    round: RoundState | None = None
    history: list[dict] = field(default_factory=list)

    # ---------------------------------------------------------------- rounds
    def open_round(self, round_id: int, assignments: Mapping[str, Mapping[str, np.ndarray]]):
        """Assign each client its OWN parent model (personalised federation)."""
        missing = set(self.clients) - set(assignments)
        if missing:
            raise ValueError(f"no parent model assigned for {sorted(missing)}")
        self.models = {c: dict(assignments[c]) for c in self.clients}
        self.round = RoundState(round_id=round_id,
                                assigned={c: state_version(self.models[c]) for c in self.clients})
        return self.round

    def parent_for(self, client_id: str) -> tuple[dict, str]:
        if client_id not in self.models:
            raise UpdateRejected(Reject.UNKNOWN_CLIENT, client_id)
        return self.models[client_id], self.round.assigned[client_id]

    # ---------------------------------------------------------------- updates
    def submit(self, env: Envelope, state: Mapping[str, np.ndarray] | None,
               authenticated_id: str) -> str:
        """Validate and record one update. `authenticated_id` comes from the
        transport (certificate in the TLS deployment) - the body's client_id is
        never trusted on its own."""
        r = self.round
        if r is None or r.closed:
            raise UpdateRejected(Reject.ROUND, "no open round")
        if env.protocol_version != PROTOCOL_VERSION:
            raise UpdateRejected(Reject.PROTOCOL, env.protocol_version)
        if env.experiment_id != self.experiment_id:
            raise UpdateRejected(Reject.EXPERIMENT, env.experiment_id)
        if env.client_id != authenticated_id:
            raise UpdateRejected(Reject.IDENTITY,
                                 f"body={env.client_id} authenticated={authenticated_id}")
        if env.client_id not in self.clients:
            raise UpdateRejected(Reject.UNKNOWN_CLIENT, env.client_id)
        if env.round_id != r.round_id:
            raise UpdateRejected(Reject.ROUND, f"round {env.round_id} != {r.round_id}")
        if state is None:
            raise UpdateRejected(Reject.MALFORMED, "no model state")
        if env.parent_version != r.assigned[env.client_id]:
            raise UpdateRejected(Reject.PARENT,
                                 f"expected {r.assigned[env.client_id]}, got {env.parent_version}")
        if env.schema_id != self.schema or schema_id(state) != self.schema:
            raise UpdateRejected(Reject.SCHEMA, env.schema_id)
        if env.update_id in r.accepted_ids:
            raise UpdateRejected(Reject.DUPLICATE, env.update_id)

        n = env.meta.get("n_train")
        if not isinstance(n, int) or n <= 0:
            raise UpdateRejected(Reject.MALFORMED, "declared sample count missing or invalid")
        r.accepted_ids.add(env.update_id)
        r.updates[env.client_id] = dict(state)
        r.counts[env.client_id] = n
        sig = env.meta.get("signature")
        if isinstance(sig, dict):
            # aggregate mask statistics only; a score-based method needs them on
            # the server. Anything else in meta is ignored.
            r.signatures[env.client_id] = sig
        return env.update_id

    def missing(self) -> list[str]:
        return [c for c in self.clients if c not in self.round.updates]

    def check_deadline(self) -> None:
        r = self.round
        if r.closed or not self.missing():
            return
        if time.monotonic() - r.opened_at > self.round_timeout_s:
            who = self.missing()
            if self.on_timeout == "pause":
                raise RoundTimeout(f"round {r.round_id} paused: still waiting for {who}")
            raise RoundTimeout(f"round {r.round_id} failed: {who} did not report")

    # ---------------------------------------------------------------- close
    def close_round(self, weights_for: Callable[[str, dict[str, int]], tuple[np.ndarray, float]]):
        """Require ALL configured clients, then aggregate one personalised model
        per client. `weights_for(client, counts) -> (donor weights, gamma)` keeps
        scoring and weighting outside the server's transport concerns."""
        r = self.round
        if self.missing():
            raise RoundTimeout(f"round {r.round_id}: incomplete, missing {self.missing()}")
        order = list(self.clients)
        states = [r.updates[c] for c in order]
        new_models = {}
        record = []
        for i, c in enumerate(order):
            w, gamma = weights_for(c, dict(r.counts))
            new_models[c] = self.aggregator.combine(states, i, np.asarray(w, float), float(gamma))
            record.append({"round": r.round_id, "receiver": c, "gamma": float(gamma),
                           "weights": {order[j]: float(w[j]) for j in range(len(order)) if j != i},
                           "parent_version": r.assigned[c],
                           "new_version": state_version(new_models[c])})
        r.closed = True
        self.history.append({"round": r.round_id, "participants": order, "aggregation": record})
        self.models = new_models
        return new_models, record
