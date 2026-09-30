"""Checkpointing and recovery for the round protocol.

Recovery policy (documented, and enforced by tests):

  1. A checkpoint is written ATOMICALLY (temp file + os.replace). An interrupted
     write leaves the previous checkpoint intact, never a half-written one.
  2. A round is either OPEN or COMPLETE, and the checkpoint says which. A
     resumed process must not confuse the two.
       * resuming an OPEN round    -> re-open it, keep the accepted update ids,
                                      and wait for the clients still missing;
       * resuming a COMPLETE round -> start the NEXT round from the aggregated
                                      personalised models.
  3. Accepted update ids are part of the checkpoint, so an update already
     applied before the interruption is rejected as a duplicate after resume.
     No update is ever applied twice.
  4. Personalised model states are saved per client (each client may hold a
     different model), as .npz arrays - never pickle. For an OPEN round the
     UPDATES ALREADY RECEIVED are saved as well: keeping only their ids would
     reject every resend as a duplicate while having lost the payloads, so the
     round could never complete.
  5. OPTIMIZER STATE IS DELIBERATELY NOT SAVED. The learner constructs a fresh
     Adam optimizer on every local training call, so optimizer state does not
     survive a call even without an interruption. Persisting it would CHANGE the
     algorithm. The reset is recorded explicitly in the checkpoint so the
     behaviour is visible rather than assumed.
  6. Per-client RNG is derived from (seed stream, round, client), so training
     randomness is reproducible from the recorded seeds; no opaque generator
     state is stored.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mechfedgnn.artifacts import atomic_write_bytes, atomic_write_json
from mechfedgnn.protocol import state_version

CHECKPOINT_SCHEMA = 1
OPTIMIZER_POLICY = {
    "persisted": False,
    "policy": "reset-per-local-training-call",
    "reason": "the learner builds a fresh Adam optimizer on every train() call; "
              "persisting optimizer state would change the algorithm",
}


@dataclass
class Checkpoint:
    experiment_id: str
    round_id: int
    round_status: str                 # "open" | "complete"
    clients: list
    assigned: dict                    # client -> parent version it was given
    accepted_update_ids: list
    counts: dict                      # client -> declared sample count
    model_schema_id: str
    config_digest: str
    seed_streams: dict
    models: dict                      # client -> parent (open) or aggregated (complete) state
    updates: dict                     # client -> update already received this round
    signatures: dict = field(default_factory=dict)   # client -> aggregate summary, if sent

    def header(self) -> dict:
        return {"schema_version": CHECKPOINT_SCHEMA, "experiment_id": self.experiment_id,
                "round_id": self.round_id, "round_status": self.round_status,
                "clients": list(self.clients), "assigned": dict(self.assigned),
                "accepted_update_ids": sorted(self.accepted_update_ids),
                "counts": dict(self.counts), "signatures": dict(self.signatures),
                "model_schema_id": self.model_schema_id,
                "config_digest": self.config_digest, "seed_streams": dict(self.seed_streams),
                "optimizer": OPTIMIZER_POLICY,
                "model_versions": {c: state_version(s) for c, s in self.models.items()},
                "update_versions": {c: state_version(s) for c, s in self.updates.items()}}


def save(directory: str, cp: Checkpoint) -> str:
    """Atomic: every model file first, then the header last. The header is the
    commit point - a checkpoint without it is ignored on resume."""
    os.makedirs(directory, exist_ok=True)
    import io

    def _dump(prefix, mapping):
        for client, state in mapping.items():
            buf = io.BytesIO()
            np.savez(buf, **{k: np.asarray(v) for k, v in state.items()})
            atomic_write_bytes(os.path.join(directory, f"{prefix}_{client}.npz"), buf.getvalue())

    _dump("model", cp.models)
    _dump("update", cp.updates)
    path = os.path.join(directory, "checkpoint.json")
    atomic_write_json(path, cp.header())
    return path


def load(directory: str) -> Checkpoint | None:
    path = os.path.join(directory, "checkpoint.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    if h.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema {h.get('schema_version')}")
    def _read(prefix, names, required):
        out = {}
        for client in names:
            p = os.path.join(directory, f"{prefix}_{client}.npz")
            if not os.path.exists(p):
                if required:
                    raise FileNotFoundError(f"checkpoint is missing {prefix} for {client}")
                continue
            with np.load(p, allow_pickle=False) as z:
                out[client] = {k: np.array(z[k]) for k in z.files}
        return out

    models = _read("model", h["clients"], required=True)
    updates = _read("update", list(h.get("update_versions", {})), required=True)
    for prefix, got_map, want_map in (("model", models, h.get("model_versions", {})),
                                      ("update", updates, h.get("update_versions", {}))):
        for client, want in want_map.items():
            got = state_version(got_map[client])
            if got != want:
                raise ValueError(f"{prefix} for {client} does not match the checkpoint "
                                 f"(expected {want}, got {got})")
    return Checkpoint(experiment_id=h["experiment_id"], round_id=h["round_id"],
                      round_status=h["round_status"], clients=h["clients"],
                      assigned=h["assigned"], accepted_update_ids=h["accepted_update_ids"],
                      counts=h["counts"], signatures=h.get("signatures") or {},
                      model_schema_id=h["model_schema_id"],
                      config_digest=h["config_digest"], seed_streams=h["seed_streams"],
                      models=models, updates=updates)


def from_coordinator(coord, config_digest: str, seed_streams: Mapping[str, Any]) -> Checkpoint:
    r = coord.round
    return Checkpoint(experiment_id=coord.experiment_id, round_id=r.round_id,
                      round_status="complete" if r.closed else "open",
                      clients=list(coord.clients), assigned=dict(r.assigned),
                      accepted_update_ids=sorted(r.accepted_ids), counts=dict(r.counts),
                      signatures=dict(r.signatures),
                      model_schema_id=coord.schema, config_digest=config_digest,
                      seed_streams=dict(seed_streams), models=dict(coord.models),
                      updates={} if r.closed else dict(r.updates))


def restore(coord, cp: Checkpoint) -> tuple[int, str]:
    """Put a coordinator back where it was. Returns (round to run next, status).

    An OPEN round is resumed with its accepted update ids intact, so a client
    that already reported cannot be applied twice. A COMPLETE round advances.
    """
    if cp.model_schema_id != coord.schema:
        raise ValueError("checkpoint model schema does not match this coordinator")
    if cp.round_status == "complete":
        coord.models = dict(cp.models)
        return cp.round_id + 1, "complete"
    coord.open_round(cp.round_id, cp.models)
    coord.round.assigned = dict(cp.assigned)          # keep the ORIGINAL assignment
    coord.round.accepted_ids = set(cp.accepted_update_ids)
    coord.round.counts = dict(cp.counts)
    coord.round.signatures = dict(cp.signatures)
    coord.round.updates = dict(cp.updates)            # payloads, not just their ids
    return cp.round_id, "open"
