"""Configuration loading, validation and resolution.

Explicit defaults, explicit types, explicit ranges. A run fails loudly on a bad
config instead of silently doing something else. The FULLY RESOLVED config
(after defaults and per-dataset overrides) is what gets recorded in provenance.

Deliberately dependency-free: no new library for a ~100-line schema.
"""
import copy
from typing import Any, Mapping

import yaml

# name -> (type(s), validator or None, default). None default = required.
SCHEMA: dict[str, tuple] = {
    "clients.K": (int, lambda v: v >= 2, None),
    "clients.G": (int, lambda v: v >= 1, 2),
    "clients.split": (list, lambda v: len(v) == 3 and abs(sum(v) - 1) < 1e-9, [0.6, 0.2, 0.2]),
    "clients.population": (str, lambda v: v in ("population_homogeneous",
                                                "population_stratified"), "population_homogeneous"),
    "model.hidden": (list, lambda v: all(int(x) > 0 for x in v), [64, 32]),
    "model.lr": (float, lambda v: v > 0, 1e-3),
    "model.batch": (int, lambda v: v >= 1, 64),
    "model.local_steps": (int, lambda v: v >= 0, None),
    "model.adapt_budget": (int, lambda v: v >= 0, None),
    "aggregation.alpha": (float, lambda v: 0.0 <= v <= 1.0, 1.0),
    "aggregation.beta": (float, lambda v: v > 0, 1.0),
    "aggregation.gamma": (float, lambda v: 0.0 <= v <= 1.0, 0.5),
    "aggregation.lambda_pop": (float, lambda v: 0.0 <= v <= 1.0, 0.5),
    "injection.enabled": (bool, None, True),
    "injection.mechanism": (str, lambda v: v in ("cell", "mcar", "mar", "fd_mnar", "cd_mnar"), "mar"),
    "injection.jitter": (float, lambda v: 0.0 <= v < 1.0, 0.05),
}


class ConfigError(ValueError):
    pass


def _get(cfg: Mapping, dotted: str, default=...):
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def validate(cfg: Mapping) -> dict:
    """Apply defaults, check types and ranges, return the resolved config."""
    out = copy.deepcopy(dict(cfg))
    problems = []
    for key, (types, check, default) in SCHEMA.items():
        value = _get(out, key, ...)
        if value is ...:
            if default is None:
                problems.append(f"{key}: required and missing")
                continue
            _set(out, key, copy.deepcopy(default))
            value = _get(out, key)
        if types is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
            _set(out, key, value)
        if not isinstance(value, types):
            problems.append(f"{key}: expected {getattr(types, '__name__', types)}, got {type(value).__name__}")
            continue
        if check is not None and not check(value):
            problems.append(f"{key}: value {value!r} fails its constraint")
    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))
    return out


def merge_overrides(cfg: Mapping, dataset: str) -> dict:
    """Deep-merge cfg['dataset_overrides'][dataset], as loop.for_dataset does."""
    def merge(base, over):
        for k, v in over.items():
            base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) else v
        return base
    return merge(copy.deepcopy(dict(cfg)), (cfg.get("dataset_overrides") or {}).get(dataset, {}))


def load(path: str, dataset: str | None = None, **overrides) -> dict:
    """Load YAML, apply per-dataset overrides and dotted overrides, validate."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if dataset:
        cfg = merge_overrides(cfg, dataset)
    for k, v in overrides.items():
        _set(cfg, k.replace("__", "."), v)
    return validate(cfg)
