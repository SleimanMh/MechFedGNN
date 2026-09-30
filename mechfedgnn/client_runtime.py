"""Client runtime: local work plus the permitted messages, nothing more.

A client reads ONLY its own assigned data file. What it sends upward is limited
to: a model update, its declared sample count, and the aggregate signature the
selected method needs. Per-example predictions are never sent; raw features,
labels and per-row masks never leave the client.
"""
import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.protocol import Envelope, schema_id, state_version, update_id
from mechfedgnn.signature_provider import MaskSignatureProvider
from model import Standardiser


def save_client_shard(path: str, X, M, y, folds: Mapping[str, np.ndarray], roles: Mapping[str, Any],
                      client_id: str) -> str:
    """Write ONE client's permitted records. Each process is given only its own."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, X=np.asarray(X, float), M=np.asarray(M, np.int8), y=np.asarray(y, float),
             train=folds["train"], val=folds["val"], test=folds["test"],
             always_observed=np.asarray(roles["always_observed"], int),
             maskable=np.asarray(roles["maskable"], int),
             client_id=np.array([client_id]))
    return path


def preprocessing_id(shared: Mapping[str, Any] | None) -> str:
    """Identity of the preprocessing COORDINATES every client must share.

    Averaging parameters only means something if the clients' inputs are in the
    same numerical coordinates. `None` means each client standardises on its own
    training fold, which is a legitimate protocol (E1 ran that way) but must be
    declared, because the resulting parameters are NOT in a common space.
    """
    if shared is None:
        return "per-client"
    blob = json.dumps({"mu": [float(v) for v in shared["mu"]],
                       "sd": [float(v) for v in shared["sd"]],
                       "y_mu": float(shared["y_mu"]), "y_sd": float(shared["y_sd"])},
                      sort_keys=True)
    return "shared:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def shared_coordinates(X, M, y) -> dict:
    """Fit ONE set of standardisation coordinates, to be handed to every client.

    Mirrors `loop.shared_scaler` (the §16 correction): fitted once, off the
    design rows, and frozen. Returned as plain JSON so it can be distributed.
    """
    st = Standardiser(np.asarray(X, float), np.asarray(M, np.int8), np.asarray(y, float))
    return {"mu": st.mu.tolist(), "sd": st.sd.tolist(),
            "y_mu": float(st.y_mu), "y_sd": float(st.y_sd)}


def _standardiser_from(shared: Mapping[str, Any], y_train) -> Standardiser:
    """Shared feature/target coordinates, the client's own AUC median.

    Mirrors `loop.scaler_for`: the coordinates are shared so parameters are
    comparable; `y_median` only binarises for AUC and stays per client.
    """
    st = Standardiser.__new__(Standardiser)
    st.mu = np.asarray(shared["mu"], float)
    st.sd = np.asarray(shared["sd"], float)
    st.y_mu, st.y_sd = float(shared["y_mu"]), float(shared["y_sd"])
    st.y_median = float(np.median(y_train))
    return st


@dataclass
class ClientRuntime:
    client_id: str
    experiment_id: str
    shard_path: str
    transport: Any
    learner: MaskAwareMLP = field(default_factory=MaskAwareMLP)
    local_steps: int = 50
    seed: int = 0
    bin_edges: Mapping[str, Any] = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    shared_scale: Mapping[str, Any] | None = None
    _data: dict = field(default_factory=dict, init=False)

    def load(self) -> dict:
        """Load this client's own shard. A client never opens another's file."""
        with np.load(self.shard_path, allow_pickle=False) as z:
            self._data = {k: np.array(z[k]) for k in z.files if k != "client_id"}
            stored = str(np.load(self.shard_path, allow_pickle=False)["client_id"][0])
        if stored != self.client_id:
            raise PermissionError(f"shard belongs to {stored}, not {self.client_id}")
        d = self._data
        tr = d["train"]
        # Shared coordinates when the experiment supplies them; otherwise the
        # client's own training fold. The choice travels with every update as
        # `preprocessing_id`, and the coordinator refuses to average across a
        # disagreement - parameters fitted in different coordinates are not
        # comparable, so averaging them is meaningless rather than merely noisy.
        self.transform = (_standardiser_from(self.shared_scale, d["y"][tr])
                          if self.shared_scale is not None
                          else Standardiser(d["X"][tr], d["M"][tr], d["y"][tr]))
        return d

    @property
    def preprocessing_id(self) -> str:
        return preprocessing_id(self.shared_scale)

    @property
    def n_train(self) -> int:
        return int(len(self._data["train"]))

    def aggregate_signature(self) -> dict:
        """Aggregate mask statistics only - never per-row masks."""
        d = self._data
        tr = d["train"]
        client = {"train": np.arange(len(tr)), "X": d["X"][tr], "M": d["M"][tr], "y": d["y"][tr]}
        roles = {"always_observed": d["always_observed"].tolist(),
                 "features": self.feature_names or [f"f{i}" for i in range(d["X"].shape[1])],
                 "bin_edges": self.bin_edges}
        summary = MaskSignatureProvider().summarize(client, roles)
        return MaskSignatureProvider().aggregate_only(summary)

    def fetch_parent(self, round_id: int) -> tuple[dict, str]:
        env = Envelope(experiment_id=self.experiment_id, client_id=self.client_id,
                       round_id=round_id, payload_type="parent_request")
        status, resp = self.transport.call("/v1/parent", env)
        if status != 200:
            raise RuntimeError(f"parent fetch failed: {resp}")
        parent_env, state = resp
        return state, parent_env.parent_version

    def local_update(self, parent: Mapping[str, np.ndarray], round_id: int) -> dict:
        d = self._data
        tr = d["train"]
        xin = self.transform.inputs(d["X"][tr], d["M"][tr])
        yt = self.transform.target(d["y"][tr])
        return self.learner.train(parent, d["X"].shape[1], xin, yt, self.local_steps,
                                  seed=self.seed + round_id)

    def submit(self, round_id: int, state: Mapping[str, np.ndarray], parent_version: str,
               schema: str) -> tuple[int, Any]:
        env = Envelope(experiment_id=self.experiment_id, client_id=self.client_id,
                       round_id=round_id, payload_type="model_update",
                       parent_version=parent_version, schema_id=schema,
                       update_id=update_id(self.client_id, round_id, state),
                       meta={"n_train": self.n_train,
                             "preprocessing_id": self.preprocessing_id})
        return self.transport.call("/v1/update", env, state)

    def run_round(self, round_id: int, schema: str) -> tuple[int, Any]:
        parent, version = self.fetch_parent(round_id)
        new_state = self.local_update(parent, round_id)
        return self.submit(round_id, new_state, version, schema)
