"""One typed message protocol, used identically in-process and over the network.

Every message carries:
  protocol_version, experiment_id, client_id (AUTHENTICATED - see transport),
  round_id, parent_version (the model version the sender expects to build on),
  schema_id (model/feature schema), payload_type, update_id (duplicate detection).

Serialization is JSON header + .npz arrays. Nothing is pickled, so a received
payload can never execute code.

The client_id inside a body is NEVER trusted on its own: the transport supplies
an authenticated identity and the server checks the two agree.
"""
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

PROTOCOL_VERSION = "1.0"
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024      # hard cap; enforced before parsing


class ProtocolError(ValueError):
    """Malformed or unacceptable message. Never leaks payload contents."""


@dataclass
class Envelope:
    experiment_id: str
    client_id: str
    round_id: int
    payload_type: str
    parent_version: str = ""
    schema_id: str = ""
    update_id: str = ""
    protocol_version: str = PROTOCOL_VERSION
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Envelope":
        required = ("experiment_id", "client_id", "round_id", "payload_type")
        missing = [k for k in required if k not in d]
        if missing:
            raise ProtocolError(f"envelope missing fields: {missing}")
        try:
            return Envelope(
                experiment_id=str(d["experiment_id"]), client_id=str(d["client_id"]),
                round_id=int(d["round_id"]), payload_type=str(d["payload_type"]),
                parent_version=str(d.get("parent_version", "")),
                schema_id=str(d.get("schema_id", "")), update_id=str(d.get("update_id", "")),
                protocol_version=str(d.get("protocol_version", "")),
                meta=dict(d.get("meta") or {}))
        except (TypeError, ValueError) as e:
            raise ProtocolError(f"envelope field has the wrong type: {e}") from None


def schema_id(state: Mapping[str, np.ndarray]) -> str:
    """Stable id over tensor names, shapes and dtypes - not values."""
    spec = [[k, list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in sorted(state.items())]
    return hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:16]


def state_version(state: Mapping[str, np.ndarray]) -> str:
    """Content version of a model state; used as parent_version."""
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(v, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def encode(env: Envelope, state: Mapping[str, np.ndarray] | None = None) -> bytes:
    """header-length | header JSON | optional .npz body."""
    buf = b""
    if state is not None:
        bio = io.BytesIO()
        np.savez(bio, **{k: np.asarray(v) for k, v in state.items()})
        buf = bio.getvalue()
    head = json.dumps(env.as_dict(), sort_keys=True).encode()
    return len(head).to_bytes(4, "big") + head + buf


def decode(blob: bytes, max_bytes: int = MAX_PAYLOAD_BYTES) -> tuple[Envelope, dict | None]:
    """Parse and validate structure. Rejects oversized, truncated or non-finite
    payloads BEFORE anything reaches aggregation."""
    if len(blob) > max_bytes:
        raise ProtocolError(f"payload too large: {len(blob)} > {max_bytes} bytes")
    if len(blob) < 4:
        raise ProtocolError("payload truncated")
    n = int.from_bytes(blob[:4], "big")
    if n <= 0 or 4 + n > len(blob):
        raise ProtocolError("header length is invalid")
    try:
        env = Envelope.from_dict(json.loads(blob[4:4 + n].decode()))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ProtocolError("header is not valid JSON") from None
    body = blob[4 + n:]
    if not body:
        return env, None
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:   # never unpickle
            state = {k: np.array(z[k]) for k in z.files}
    except Exception:
        raise ProtocolError("body is not a readable .npz array archive") from None
    for k, v in state.items():
        if not np.issubdtype(v.dtype, np.floating):
            raise ProtocolError(f"tensor '{k}' is not floating point")
        if not np.isfinite(v).all():
            raise ProtocolError(f"tensor '{k}' contains non-finite values")
    return env, state


def update_id(client_id: str, round_id: int, state: Mapping[str, np.ndarray]) -> str:
    """Deterministic per (client, round, content): a retry of the SAME update
    carries the same id and must be accepted at most once."""
    return hashlib.sha256(f"{client_id}|{round_id}|{state_version(state)}".encode()).hexdigest()[:16]
