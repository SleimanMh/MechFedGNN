"""Artifact store: configs, metrics, weights, checkpoints and provenance.

Rules this store enforces:
  * a COMPLETED run is never silently overwritten (a marker file records
    completion; writing again requires an explicit new run id);
  * every write is ATOMIC (temp file + os.replace), so an interrupted write
    cannot leave a half-file that later looks valid;
  * raw input files are never modified.
"""
import json
import os
import tempfile
from typing import Any, Mapping

import numpy as np

DONE = "_COMPLETE.json"


class RunExists(FileExistsError):
    pass


def atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                 # atomic on POSIX and Windows
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path: str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True, default=_default))


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


class FileArtifactStore:
    def __init__(self, root: str = "results"):
        self.root = root

    def run_dir(self, experiment: str, run_id: str) -> str:
        return os.path.join(self.root, experiment, run_id)

    def is_complete(self, experiment: str, run_id: str) -> bool:
        return os.path.exists(os.path.join(self.run_dir(experiment, run_id), DONE))

    def begin(self, experiment: str, run_id: str, manifest: Mapping[str, Any]) -> str:
        """Create the run directory and write provenance. Refuses to touch a run
        that is already marked complete."""
        d = self.run_dir(experiment, run_id)
        if self.is_complete(experiment, run_id):
            raise RunExists(f"run already complete: {d} (use a new run id)")
        os.makedirs(d, exist_ok=True)
        atomic_write_json(os.path.join(d, "provenance.json"), manifest)
        return d

    def write_table(self, experiment: str, run_id: str, name: str, rows) -> str | None:
        import pandas as pd
        if rows is None or (hasattr(rows, "__len__") and len(rows) == 0):
            return None
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        path = os.path.join(self.run_dir(experiment, run_id), f"{name}.csv")
        atomic_write_text(path, df.to_csv(index=False, float_format="%.10g"))
        return path

    def write_state(self, experiment: str, run_id: str, name: str,
                    state: Mapping[str, np.ndarray]) -> str:
        """Model parameters as .npz - arrays only, never pickled objects, so a
        stored state cannot execute code when loaded."""
        import io
        buf = io.BytesIO()
        np.savez(buf, **{k: np.asarray(v) for k, v in state.items()})
        path = os.path.join(self.run_dir(experiment, run_id), f"{name}.npz")
        atomic_write_bytes(path, buf.getvalue())
        return path

    def complete(self, experiment: str, run_id: str, summary: Mapping[str, Any]) -> str:
        path = os.path.join(self.run_dir(experiment, run_id), DONE)
        atomic_write_json(path, dict(summary))
        return path
