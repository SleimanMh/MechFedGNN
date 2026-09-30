"""Milestone E: checkpointing and recovery.

Verifies the documented policy in mechfedgnn/checkpoint.py: atomic writes, a
complete round distinguished from an incomplete one, no update applied twice
after resume, and an explicitly documented optimizer reset.
"""
import json
import os

import numpy as np
import pytest

from mechfedgnn.checkpoint import (CHECKPOINT_SCHEMA, OPTIMIZER_POLICY, from_coordinator, load,
                                   restore, save)
from mechfedgnn.protocol import Envelope, schema_id, state_version, update_id
from mechfedgnn.server import Reject, ServerCoordinator, UpdateRejected

CLIENTS = ["c0", "c1", "c2"]
EXP = "demo"
STREAMS = {"partition": 1, "mask": 2, "train": 3, "mode": "independent"}


def tiny_state(scale=1.0):
    return {"w": np.arange(4.0) * scale, "b": np.array([0.5 * scale])}


def fresh_coord():
    c = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema=schema_id(tiny_state()))
    c.open_round(1, {cid: tiny_state(1 + i) for i, cid in enumerate(CLIENTS)})
    return c


def env_for(coord, cid, state, round_id=None):
    r = round_id or coord.round.round_id
    return Envelope(experiment_id=EXP, client_id=cid, round_id=r, payload_type="model_update",
                    parent_version=coord.round.assigned[cid], schema_id=coord.schema,
                    update_id=update_id(cid, r, state), meta={"n_train": 100})


def uniform(k, i):
    w = np.full(k, 1.0 / (k - 1))
    w[i] = 0.0
    return w


def test_checkpoint_round_trips_and_records_the_optimizer_policy(tmp_path):
    coord = fresh_coord()
    coord.submit(env_for(coord, "c0", tiny_state(9)), tiny_state(9), "c0")
    cp = from_coordinator(coord, config_digest="cfg123", seed_streams=STREAMS)
    save(str(tmp_path), cp)

    header = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert header["schema_version"] == CHECKPOINT_SCHEMA
    assert header["round_status"] == "open"
    assert header["optimizer"] == OPTIMIZER_POLICY
    assert header["optimizer"]["persisted"] is False          # deliberate, documented
    assert header["seed_streams"] == STREAMS
    assert header["config_digest"] == "cfg123"

    back = load(str(tmp_path))
    assert back.round_status == "open" and back.round_id == 1
    assert back.accepted_update_ids == sorted(coord.round.accepted_ids)
    for c in CLIENTS:
        np.testing.assert_array_equal(back.models[c]["w"], coord.models[c]["w"])
    np.testing.assert_array_equal(back.updates["c0"]["w"], tiny_state(9)["w"])


def test_interrupted_open_round_resumes_without_applying_an_update_twice(tmp_path):
    """Interrupt at a controlled boundary: two clients have reported, one has not."""
    coord = fresh_coord()
    s0, s1 = tiny_state(9), tiny_state(8)
    coord.submit(env_for(coord, "c0", s0), s0, "c0")
    coord.submit(env_for(coord, "c1", s1), s1, "c1")
    assert coord.missing() == ["c2"]
    save(str(tmp_path), from_coordinator(coord, "cfg", STREAMS))

    # ---- process dies here; a brand new coordinator comes up ----
    revived = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema=schema_id(tiny_state()))
    cp = load(str(tmp_path))
    next_round, status = restore(revived, cp)
    assert (next_round, status) == (1, "open")
    assert revived.missing() == ["c2"]                        # accounting preserved

    # c0 retries the SAME update it already sent: must not be applied twice
    with pytest.raises(UpdateRejected) as e:
        revived.submit(env_for(revived, "c0", s0), s0, "c0")
    assert e.value.reason is Reject.DUPLICATE
    assert len(revived.round.updates) == 2

    # the missing client reports and the round completes normally
    s2 = tiny_state(7)
    revived.submit(env_for(revived, "c2", s2), s2, "c2")
    models, record = revived.close_round(lambda c, n: (uniform(3, CLIENTS.index(c)), 0.5))
    assert set(models) == set(CLIENTS) and len(record) == 3


def test_completed_round_resumes_at_the_next_round(tmp_path):
    coord = fresh_coord()
    for i, c in enumerate(CLIENTS):
        st = tiny_state(5 + i)
        coord.submit(env_for(coord, c, st), st, c)
    models, _ = coord.close_round(lambda c, n: (uniform(3, CLIENTS.index(c)), 0.5))
    save(str(tmp_path), from_coordinator(coord, "cfg", STREAMS))

    header = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert header["round_status"] == "complete"

    revived = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema=schema_id(tiny_state()))
    next_round, status = restore(revived, load(str(tmp_path)))
    assert (next_round, status) == (2, "complete")
    for c in CLIENTS:                                          # aggregated models carried over
        np.testing.assert_allclose(revived.models[c]["w"], models[c]["w"])


def test_resumed_completed_round_does_not_reapply_its_updates(tmp_path):
    """A completed round must not be re-aggregated on resume."""
    coord = fresh_coord()
    for i, c in enumerate(CLIENTS):
        st = tiny_state(5 + i)
        coord.submit(env_for(coord, c, st), st, c)
    models, _ = coord.close_round(lambda c, n: (uniform(3, CLIENTS.index(c)), 0.5))
    versions = {c: state_version(models[c]) for c in CLIENTS}
    save(str(tmp_path), from_coordinator(coord, "cfg", STREAMS))

    revived = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema=schema_id(tiny_state()))
    restore(revived, load(str(tmp_path)))
    assert {c: state_version(revived.models[c]) for c in CLIENTS} == versions


def test_tampered_or_missing_model_file_is_detected(tmp_path):
    coord = fresh_coord()
    save(str(tmp_path), from_coordinator(coord, "cfg", STREAMS))
    np.savez(tmp_path / "model_c1.npz", w=np.zeros(4), b=np.zeros(1))   # tampered
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        load(str(tmp_path))

    os.remove(tmp_path / "model_c0.npz")
    with pytest.raises((FileNotFoundError, ValueError)):
        load(str(tmp_path))


def test_schema_mismatch_on_restore_is_refused(tmp_path):
    coord = fresh_coord()
    save(str(tmp_path), from_coordinator(coord, "cfg", STREAMS))
    other = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema="a-different-schema")
    with pytest.raises(ValueError, match="schema"):
        restore(other, load(str(tmp_path)))


def test_missing_checkpoint_returns_none_not_an_error(tmp_path):
    assert load(str(tmp_path)) is None


def test_checkpoint_write_is_atomic_and_leaves_no_partial_files(tmp_path):
    coord = fresh_coord()
    save(str(tmp_path), from_coordinator(coord, "cfg", STREAMS))
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "checkpoint.json").exists()
    # the header is the commit point: without it a resume finds nothing
    os.remove(tmp_path / "checkpoint.json")
    assert load(str(tmp_path)) is None
