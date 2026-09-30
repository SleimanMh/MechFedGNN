"""Run provenance: everything needed to say what produced a result.

Records code revision AND whether the tree was dirty, the fully resolved
config, dataset identity and checksum, split / duplicate-group / mask
identifiers, the preprocessing definition and WHERE IT WAS FITTED, the seed
streams, model architecture and tensor schema, budgets, and (filled by the
runner) scores, realised weights, fallbacks, participation and metrics.

Nothing here reads client-level features or labels: identifiers are hashes.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


def _run(args: Sequence[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True,
                              timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_revision() -> dict:
    """Commit, dirty flag and branch. A dirty tree makes a run non-reproducible
    from the commit alone, so it is recorded explicitly, never inferred."""
    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": sha, "short": (sha or "")[:7] or None,
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": None if status is None else bool(status.strip()),
            "dirty_files": [] if not status else sorted(
                l[3:] for l in status.splitlines() if l[3:])[:50]}


def sha_of_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


def dataset_identity(name: str, df, path: str | None = None) -> dict:
    """Identity and checksum of the table actually used (after any column
    drops), so a silently changed raw file is detectable."""
    import pandas as pd
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()
    return {"name": name, "path": path, "n_rows": int(len(df)),
            "columns": list(map(str, df.columns)), "sha256": digest[:32],
            "file_sha256": file_sha(path) if path and os.path.exists(path) else None}


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def split_identity(clients: Sequence[Mapping[str, Any]], dup_groups=None) -> dict:
    """Identifiers for the partition, the folds and the masks - hashes only."""
    out = {"n_clients": len(clients), "clients": {}}
    for c in clients:
        out["clients"][c["id"]] = {
            "n_rows": int(len(c["rows"])),
            "rows_sha": sha_of_array(c["rows"]),
            "folds": {f: {"n": int(len(c[f])), "sha": sha_of_array(c[f])}
                      for f in ("train", "val", "test") if f in c},
            "mask_sha": None if c.get("M") is None else sha_of_array(c["M"]),
            "mask_missing_per_feature": None if c.get("M") is None
            else (c["M"] == 0).sum(0).tolist(),
        }
    if dup_groups is not None:
        out["duplicate_groups"] = {
            "n_groups": int(len(np.unique(dup_groups))),
            "n_rows_in_a_group": int((np.bincount(dup_groups)[dup_groups] > 1).sum()),
            "groups_sha": sha_of_array(dup_groups)}
    return out


def preprocessing_identity(kind: str, fitted_on: str, transform=None) -> dict:
    """WHERE the transform was fitted is part of the record, not an assumption."""
    out = {"kind": kind, "fitted_on": fitted_on}
    if transform is not None:
        out["feature_centre_sha"] = sha_of_array(np.asarray(transform.mu, float))
        out["feature_scale_sha"] = sha_of_array(np.asarray(transform.sd, float))
        out["target_centre"] = float(transform.y_mu)
        out["target_scale"] = float(transform.y_sd)
    return out


def environment() -> dict:
    import numpy
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": numpy.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    return env


def manifest(*, experiment: str, dataset: dict, config: Mapping[str, Any], seeds: Mapping[str, Any],
             splits: dict, preprocessing: dict, model: dict, budgets: dict,
             injection: dict) -> dict:
    """The provenance block written beside every run."""
    return {"experiment": experiment, "code": code_revision(), "environment": environment(),
            "dataset": dataset, "config_resolved": json.loads(json.dumps(config, default=str)),
            "seed_streams": dict(seeds), "splits": splits, "preprocessing": preprocessing,
            "model": model, "budgets": budgets, "injection": injection}
