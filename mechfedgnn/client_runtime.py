"""Client runtime: local work plus the permitted messages, nothing more.

A client reads ONLY its own assigned data file. What it sends upward is limited
to: a model update, its declared sample count, and the aggregate signature the
selected method needs. Per-example predictions are never sent; raw features,
labels and per-row masks never leave the client.
"""
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
        self.transform = Standardiser(d["X"][tr], d["M"][tr], d["y"][tr])
        return d

    @property
    def n_train(self) -> int:
        return int(len(self._data["train"]))

    @staticmethod
    def _json_safe(x):
        """Aggregate statistics as plain JSON. Shapes are per-FEATURE, never per-row."""
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: ClientRuntime._json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [ClientRuntime._json_safe(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

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
               schema: str, send_signature: bool = False) -> tuple[int, Any]:
        """`send_signature` attaches the AGGREGATE mask summary, which is what a
        score-based method needs on the server. It is opt-in because a method
        that does not use scores must not transmit more than it needs; what it
        attaches is exactly `MaskSignatureProvider.aggregate_only`, so per-row
        masks, features and labels still cannot leave."""
        meta = {"n_train": self.n_train}
        if send_signature:
            meta["signature"] = self._json_safe(self.aggregate_signature())
        env = Envelope(experiment_id=self.experiment_id, client_id=self.client_id,
                       round_id=round_id, payload_type="model_update",
                       parent_version=parent_version, schema_id=schema,
                       update_id=update_id(self.client_id, round_id, state),
                       meta=meta)
        return self.transport.call("/v1/update", env, state)

    def run_round(self, round_id: int, schema: str, send_signature: bool = False) -> tuple[int, Any]:
        parent, version = self.fetch_parent(round_id)
        new_state = self.local_update(parent, round_id)
        return self.submit(round_id, new_state, version, schema, send_signature)
