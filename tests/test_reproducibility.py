"""Milestone B: validated config, full provenance, separate RNG streams,
optional injection, and an artifact store that never silently overwrites."""
import copy
import json
import os

import numpy as np
import pytest

from data.inject import build_roles
from mechfedgnn.artifacts import FileArtifactStore, RunExists, atomic_write_json
from mechfedgnn.config import ConfigError, validate
from mechfedgnn.provenance import code_revision, dataset_identity, split_identity
from mechfedgnn.runner import execute, natural_missingness_clients
from mechfedgnn.seeds import SeedBundle
from tools.capture_reference import base_cfg, synthetic_df


# ---------------------------------------------------------------- config

def test_validate_applies_defaults_and_accepts_a_good_config():
    cfg = validate({"clients": {"K": 4}, "model": {"local_steps": 10, "adapt_budget": 2}})
    assert cfg["clients"]["split"] == [0.6, 0.2, 0.2]          # default applied
    assert cfg["aggregation"]["gamma"] == 0.5
    assert cfg["injection"]["enabled"] is True


@pytest.mark.parametrize("bad, needle", [
    ({"clients": {"K": 1}}, "clients.K"),
    ({"clients": {"K": 4, "split": [0.6, 0.2, 0.9]}}, "clients.split"),
    ({"clients": {"K": 4}, "aggregation": {"gamma": 1.7}}, "aggregation.gamma"),
    ({"clients": {"K": 4}, "injection": {"mechanism": "nope"}}, "injection.mechanism"),
    ({"clients": {"K": 4}}, "model.local_steps"),              # required, missing
])
def test_validate_rejects_bad_values_loudly(bad, needle):
    base = {"model": {"local_steps": 10, "adapt_budget": 2}}
    cfg = {**base, **bad} if "model" not in bad else dict(bad)
    if needle == "model.local_steps":
        cfg = {"clients": {"K": 4}}
    with pytest.raises(ConfigError) as e:
        validate(cfg)
    assert needle in str(e.value)


# ---------------------------------------------------------------- seeds

def test_legacy_seed_bundle_reproduces_the_historical_derivation():
    """Default must be bit-identical to the research code, or every historical
    result would silently change."""
    from loop import _seed
    b = SeedBundle.from_config({}, 11)
    assert b.mode == "legacy" and (b.partition, b.mask, b.train) == (11, 11, 11)
    for parts in [(10,), (20, 0), (30, 3)]:
        assert b.derive("train", *parts) == _seed(11, *parts)


def test_streams_can_be_varied_independently():
    a = SeedBundle.from_config({"seed_streams": {"partition": 1, "mask": 2, "train": 3}}, 11)
    b = SeedBundle.from_config({"seed_streams": {"partition": 1, "mask": 99, "train": 3}}, 11)
    assert a.mode == "independent"
    assert a.derive("partition", 0) == b.derive("partition", 0)     # partition unchanged
    assert a.derive("train", 0) == b.derive("train", 0)             # training unchanged
    assert a.derive("mask", 0) != b.derive("mask", 0)               # only masks move


# ---------------------------------------------------------------- provenance

def test_provenance_records_revision_dirty_flag_and_identifiers():
    df = synthetic_df(400)
    roles = build_roles(df)
    cfg = base_cfg()
    rev = code_revision()
    assert set(rev) >= {"commit", "dirty", "branch"}
    assert rev["dirty"] in (True, False, None)                       # explicit, never inferred
    ident = dataset_identity("synthetic", df)
    assert ident["n_rows"] == len(df) and len(ident["sha256"]) == 32
    clients, _ = natural_missingness_clients(df, roles, cfg, 11)
    sp = split_identity(clients)
    assert sp["n_clients"] == len(clients)
    for c in clients:
        assert set(sp["clients"][c["id"]]["folds"]) == {"train", "val", "test"}
        assert sp["clients"][c["id"]]["mask_sha"] is not None


def test_dataset_checksum_changes_when_a_value_changes():
    df = synthetic_df(300)
    a = dataset_identity("s", df)["sha256"]
    df2 = df.copy()
    df2.iloc[0, 0] = df2.iloc[0, 0] + 1.0
    assert dataset_identity("s", df2)["sha256"] != a


# ---------------------------------------------------------------- artifact store

def test_store_is_atomic_and_refuses_to_overwrite_a_completed_run(tmp_path):
    store = FileArtifactStore(root=str(tmp_path))
    store.begin("exp", "r1", {"provenance": True})
    assert os.path.exists(os.path.join(store.run_dir("exp", "r1"), "provenance.json"))
    store.write_table("exp", "r1", "metrics", [{"a": 1}, {"a": 2}])
    store.complete("exp", "r1", {"run_id": "r1"})
    assert store.is_complete("exp", "r1")
    with pytest.raises(RunExists):
        store.begin("exp", "r1", {"provenance": True})               # no silent overwrite
    store.begin("exp", "r2", {"provenance": True})                   # a new id is fine


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(str(p), {"k": [1, 2, 3]})
    assert json.loads(p.read_text(encoding="utf-8"))["k"] == [1, 2, 3]
    assert not list(tmp_path.glob("*.tmp"))


def test_state_is_saved_as_arrays_not_pickle(tmp_path):
    store = FileArtifactStore(root=str(tmp_path))
    store.begin("exp", "r", {})
    path = store.write_state("exp", "r", "theta", {"w": np.arange(4.0)})
    with np.load(path) as z:                                          # loads without allow_pickle
        np.testing.assert_array_equal(z["w"], np.arange(4.0))


# ---------------------------------------------------------------- optional injection

def test_client_with_naturally_incomplete_data_trains_without_an_injector(tmp_path):
    """Injection is optional: the mask comes from the data's own NaN pattern."""
    df = synthetic_df(900)
    rng = np.random.default_rng(0)
    holes = rng.random(df[[c for c in df.columns if c != "target"]].shape) < 0.15
    feats = [c for c in df.columns if c != "target"]
    df[feats] = df[feats].mask(holes)
    assert df[feats].isna().to_numpy().any()

    roles = build_roles(df.fillna(0.0))
    cfg = copy.deepcopy(base_cfg())
    cfg["injection"] = {"enabled": False}
    cfg["arms"] = ["local-only", "uniform-donor", "coverage-W_H"]

    clients, _ = natural_missingness_clients(df, roles, cfg, 11)
    assert all((c["M"] == 0).any() for c in clients)                  # real missingness present

    run_dir, out = execute("t_natural", "synthetic", df.fillna(0.0), roles, cfg, 11,
                           builder=lambda s: natural_missingness_clients(df, roles, cfg, s),
                           store=FileArtifactStore(root=str(tmp_path)))
    man = json.load(open(os.path.join(run_dir, "provenance.json"), encoding="utf-8"))
    assert man["injection"]["enabled"] is False
    assert man["injection"]["source"] == "data's own NaN pattern"
    assert os.path.exists(os.path.join(run_dir, "metrics.csv"))
    assert os.path.exists(os.path.join(run_dir, "weights.csv"))
    assert len(out["metrics"]) > 0


def test_execute_records_realised_weights_fallbacks_and_participation(tmp_path):
    df = synthetic_df(900)
    roles = build_roles(df)
    cfg = copy.deepcopy(base_cfg())
    cfg["arms"] = ["local-only", "uniform-donor", "coverage-W_H", "population-S"]
    run_dir, out = execute("t_prov", "synthetic", df, roles, cfg, 11,
                           store=FileArtifactStore(root=str(tmp_path)))
    done = json.load(open(os.path.join(run_dir, "_COMPLETE.json"), encoding="utf-8"))
    assert done["n_weight_rows"] > 0 and len(done["participation"]) == cfg["clients"]["K"]
    assert isinstance(done["fallbacks"], list)
    import pandas as pd
    w = pd.read_csv(os.path.join(run_dir, "weights.csv"))
    assert {"receiver", "arm", "donor", "weight", "gamma", "fallback"} <= set(w.columns)
    man = json.load(open(os.path.join(run_dir, "provenance.json"), encoding="utf-8"))
    assert man["preprocessing"]["fitted_on"]                          # where it was fitted
    assert man["seed_streams"]["mode"] == "legacy"
    assert man["dataset"]["sha256"] and man["splits"]["n_clients"] == cfg["clients"]["K"]
