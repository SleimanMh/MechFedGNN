"""Reproducible run driver (Milestone B).

Wraps the existing orchestrator without changing its behaviour, and records
everything needed to reproduce or audit a run. Also provides the OPTIONAL
injection path: a client whose data is already incomplete trains on its own
missingness with no injector at all.
"""
import copy
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from data.clients import _split, dup_groups, relocate_duplicates, assign_rows_homogeneous
from mechfedgnn import provenance
from mechfedgnn.artifacts import FileArtifactStore
from mechfedgnn.seeds import SeedBundle


def natural_missingness_clients(df, roles, cfg, seed):
    """Clients whose mask is the data's OWN incompleteness (NaN), with no
    injector. Satisfies: a client with naturally incomplete data must be able to
    train without additional removal.

    Returns (clients, groups) in the same shape the orchestrator expects.
    """
    feats = roles["features"]
    X = df[feats].to_numpy(float)
    y = df["target"].to_numpy(float)
    M = (~np.isnan(X)).astype(np.int8)          # M = 1 observed
    X = np.nan_to_num(X, nan=0.0)
    cc = cfg["clients"]
    groups_dup = dup_groups(df, cfg)
    seeds = SeedBundle.from_config(cfg, seed)
    rng = np.random.default_rng([seeds.partition, 0])
    parts = assign_rows_homogeneous(len(X), cc["K"], rng)
    splits = [_split(len(r), cc["split"], np.random.default_rng([seeds.partition, 3, k]))
              for k, r in enumerate(parts)]
    if groups_dup is not None:
        parts, splits = relocate_duplicates(len(X), parts, splits, groups_dup)
    clients = []
    for k, rows in enumerate(parts):
        clients.append({"id": f"c{k}", "k": k, "rows": rows, "X": X[rows], "y": y[rows],
                        "M": M[rows], "profile": None, "group": k % max(cc["G"], 1),
                        **splits[k]})
    return clients, np.array([c["group"] for c in clients])


def injection_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool((cfg.get("injection") or {}).get("enabled", True))


def run_id_for(experiment: str, dataset: str, cfg: Mapping[str, Any], seeds: SeedBundle) -> str:
    """Stable id from the resolved config and seed streams; the code revision
    and dirty flag live in provenance, not in the name."""
    blob = json.dumps({"e": experiment, "d": dataset, "c": cfg, "s": seeds.as_dict()},
                      sort_keys=True, default=str)
    rev = provenance.code_revision()
    tag = (rev["short"] or "nogit") + ("-dirty" if rev["dirty"] else "")
    return f"{dataset}_{tag}_{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


def execute(experiment, dataset, df, roles, cfg, seed, *, builder=None, folds=("val", "test"),
            headroom=False, store=None, dataset_path=None, preprocessing_kind="shared-design-split"):
    """Run one seed and persist config, provenance, metrics, scores, REALISED
    weights, fallbacks and participation. Returns (run_dir, outputs)."""
    from loop import run_seed, shared_scaler

    store = store or FileArtifactStore()
    seeds = SeedBundle.from_config(cfg, seed)
    rid = run_id_for(experiment, dataset, cfg, seeds)

    if builder is None and not injection_enabled(cfg):
        builder = lambda s: natural_missingness_clients(df, roles, cfg, s)
    clients, _ = (builder(seed) if builder else
                  __import__("data.clients", fromlist=["build_clients"]).build_clients(
                      df, roles, cfg, seed))

    shared = shared_scaler(df, roles, cfg)
    man = provenance.manifest(
        experiment=experiment,
        dataset=provenance.dataset_identity(dataset, df, dataset_path),
        config=cfg, seeds=seeds.as_dict(),
        splits=provenance.split_identity(clients, dup_groups(df, cfg)),
        preprocessing=provenance.preprocessing_identity(
            preprocessing_kind if shared is not None else "per-client",
            "design-split rows" if shared is not None else "each client's own training fold",
            shared),
        model={"learner": "mask-aware-mlp", "hidden": list(cfg["model"]["hidden"]),
               "lr": cfg["model"]["lr"], "batch": cfg["model"]["batch"],
               "n_features": len(roles["features"])},
        budgets={"local_steps": cfg["model"]["local_steps"],
                 "adapt_budget": cfg["model"]["adapt_budget"]},
        injection={"enabled": injection_enabled(cfg),
                   "mechanism": (cfg.get("injection") or {}).get("mechanism"),
                   "source": "injector" if injection_enabled(cfg) else "data's own NaN pattern"})
    run_dir = store.begin(experiment, rid, man)

    out = run_seed(df, roles, copy.deepcopy(cfg), seed, folds=folds, headroom=headroom,
                   builder=builder)

    for name in ["metrics", "scores", "weights", "candidates", "compute", "geometry", "function",
                 "headroom"]:
        store.write_table(experiment, rid, name, out.get(name))
    participation = sorted({w["receiver"] for w in out["weights"]}) or [c["id"] for c in clients]
    fallbacks = sorted({(w["arm"], w["fallback"]) for w in out["weights"] if w["fallback"]})
    store.complete(experiment, rid, {
        "run_id": rid, "seed": seed, "n_clients": len(clients),
        "participation": participation,
        "arms": list(cfg.get("arms") or []),
        "fallbacks": [{"arm": a, "reason": r} for a, r in fallbacks],
        "n_metric_rows": len(out.get("metrics", [])),
        "n_weight_rows": len(out.get("weights", []))})
    return run_dir, out
