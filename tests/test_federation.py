"""Milestone C: client/server separation, protocol validation and policies.

These verify the software boundaries and the declared policies. They are not a
scientific result: the multi-round demo exists to exercise the framework.
"""
import json
import os

import numpy as np
import pytest

from mechfedgnn.aggregation import WeightedAverage
from mechfedgnn.client_runtime import ClientRuntime, save_client_shard
from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.protocol import (Envelope, PROTOCOL_VERSION, ProtocolError, decode, encode,
                                 schema_id, state_version, update_id)
from mechfedgnn.server import Reject, RoundTimeout, ServerCoordinator, UpdateRejected
from mechfedgnn.transport import HttpTransport, InProcessTransport, ServerApp, serve

CLIENTS = ["c0", "c1", "c2"]
EXP = "demo"


def tiny_state(scale=1.0):
    return {"w": np.arange(4.0) * scale, "b": np.array([0.5 * scale])}


def make_coord(**kw):
    s = tiny_state()
    c = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema=schema_id(s), **kw)
    # personalised: each client gets its OWN parent model
    c.open_round(1, {cid: tiny_state(1 + i) for i, cid in enumerate(CLIENTS)})
    return c, s


def env_for(coord, cid, state, round_id=1, **over):
    parent, version = coord.models[cid], coord.round.assigned[cid]
    base = dict(experiment_id=EXP, client_id=cid, round_id=round_id,
                payload_type="model_update", parent_version=version,
                schema_id=coord.schema, update_id=update_id(cid, round_id, state),
                meta={"n_train": 100})
    base.update(over)
    return Envelope(**base)


# ---------------------------------------------------------------- protocol

def test_envelope_round_trips_and_rejects_malformed():
    s = tiny_state()
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=2, payload_type="model_update")
    back, state = decode(encode(env, s))
    assert back.client_id == "c0" and back.round_id == 2
    np.testing.assert_array_equal(state["w"], s["w"])
    with pytest.raises(ProtocolError):
        decode(b"\x00\x00\x00\x05nope")
    with pytest.raises(ProtocolError):
        decode(encode(env, s), max_bytes=10)                       # oversized


def test_non_finite_tensor_is_rejected_before_aggregation():
    bad = {"w": np.array([1.0, np.inf, 2.0, 3.0]), "b": np.array([0.0])}
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="model_update")
    with pytest.raises(ProtocolError, match="non-finite"):
        decode(encode(env, bad))


def test_payload_cannot_carry_executable_objects():
    """np.load(..., allow_pickle=False) is what makes an uploaded state inert."""
    import io
    import pickle
    blob = pickle.dumps({"evil": 1})
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="model_update")
    head = encode(env, None)
    with pytest.raises(ProtocolError):
        decode(head + blob)


# ---------------------------------------------------------------- validation

def test_personalised_parent_is_validated_per_client():
    """Each client has its own assigned parent; c1's version must not satisfy c0."""
    coord, s = make_coord()
    assert len({v for v in coord.round.assigned.values()}) == len(CLIENTS)
    wrong = env_for(coord, "c0", s, parent_version=coord.round.assigned["c1"])
    with pytest.raises(UpdateRejected) as e:
        coord.submit(wrong, s, "c0")
    assert e.value.reason is Reject.PARENT


@pytest.mark.parametrize("over, reason", [
    (dict(protocol_version="0.9"), Reject.PROTOCOL),
    (dict(experiment_id="other"), Reject.EXPERIMENT),
    (dict(round_id=7), Reject.ROUND),
    (dict(schema_id="deadbeef"), Reject.SCHEMA),
    (dict(meta={}), Reject.MALFORMED),
])
def test_invalid_updates_are_rejected(over, reason):
    coord, s = make_coord()
    with pytest.raises(UpdateRejected) as e:
        coord.submit(env_for(coord, "c0", s, **over), s, "c0")
    assert e.value.reason is reason


def test_body_client_id_is_not_trusted():
    coord, s = make_coord()
    spoof = env_for(coord, "c0", s)                     # claims c0 ...
    with pytest.raises(UpdateRejected) as e:
        coord.submit(spoof, s, authenticated_id="c2")   # ... but authenticated as c2
    assert e.value.reason is Reject.IDENTITY


def test_duplicate_update_cannot_be_applied_twice():
    coord, s = make_coord()
    coord.submit(env_for(coord, "c0", s), s, "c0")
    with pytest.raises(UpdateRejected) as e:
        coord.submit(env_for(coord, "c0", s), s, "c0")   # identical retry
    assert e.value.reason is Reject.DUPLICATE
    assert len(coord.round.updates) == 1


def test_stale_update_for_a_closed_round_is_rejected():
    coord, s = make_coord()
    for cid in CLIENTS:
        coord.submit(env_for(coord, cid, s), s, cid)
    coord.close_round(lambda c, counts: (_uniform(len(CLIENTS), CLIENTS.index(c)), 0.5))
    with pytest.raises(UpdateRejected) as e:
        coord.submit(env_for(coord, "c0", s), s, "c0")
    assert e.value.reason is Reject.ROUND


def _uniform(k, i):
    w = np.full(k, 1.0 / (k - 1))
    w[i] = 0.0
    return w


# ---------------------------------------------------------------- policies

def test_round_requires_all_configured_clients():
    coord, s = make_coord()
    coord.submit(env_for(coord, "c0", s), s, "c0")
    assert coord.missing() == ["c1", "c2"]
    with pytest.raises(RoundTimeout, match="incomplete"):
        coord.close_round(lambda c, counts: (_uniform(3, 0), 0.5))


@pytest.mark.parametrize("policy, needle", [("fail", "failed"), ("pause", "paused")])
def test_timeout_is_explicit_and_does_not_change_participants(policy, needle):
    coord, s = make_coord(round_timeout_s=0.0, on_timeout=policy)
    coord.submit(env_for(coord, "c0", s), s, "c0")
    with pytest.raises(RoundTimeout, match=needle):
        coord.check_deadline()
    assert coord.missing() == ["c1", "c2"]               # participants untouched


def test_close_round_produces_one_personalised_model_per_client():
    coord, s = make_coord()
    for cid in CLIENTS:
        coord.submit(env_for(coord, cid, tiny_state(1 + CLIENTS.index(cid))),
                     tiny_state(1 + CLIENTS.index(cid)), cid)
    models, record = coord.close_round(
        lambda c, counts: (_nonuniform(len(CLIENTS), CLIENTS.index(c)), 0.5))
    assert set(models) == set(CLIENTS)
    versions = {r["receiver"]: r["new_version"] for r in record}
    assert len(set(versions.values())) == len(CLIENTS)   # personalised, not one global model


def _nonuniform(k, i):
    """Deliberately unequal donor weights so an aggregation or transport bug
    cannot pass merely because all weights are equal."""
    w = np.array([0.2, 0.3, 0.5][:k], dtype=float)
    w[i] = 0.0
    return w / w.sum()


# ---------------------------------------------------------------- transports

def _run_round_over(transport_factory, coord):
    app = ServerApp(coord)
    results = {}
    for cid in CLIENTS:
        t = transport_factory(app, cid)
        env = Envelope(experiment_id=EXP, client_id=cid, round_id=1, payload_type="parent_request")
        status, (penv, parent) = t.call("/v1/parent", env)
        assert status == 200
        new = {k: v * 1.1 for k, v in parent.items()}
        up = Envelope(experiment_id=EXP, client_id=cid, round_id=1, payload_type="model_update",
                      parent_version=penv.parent_version, schema_id=coord.schema,
                      update_id=update_id(cid, 1, new), meta={"n_train": 50 + 10 * CLIENTS.index(cid)})
        status, resp = t.call("/v1/update", up, new)
        assert status == 200, resp
        results[cid] = new
    models, record = coord.close_round(
        lambda c, counts: (_nonuniform(len(CLIENTS), CLIENTS.index(c)), 0.5))
    return models, record


def test_in_process_and_network_transports_produce_identical_aggregation():
    """Same logical protocol, two routes, identical result - with NON-UNIFORM
    weights so equality is not an artefact of uniformity."""
    coord_a, _ = make_coord()
    models_a, rec_a = _run_round_over(lambda app, cid: InProcessTransport(app, cid), coord_a)

    coord_b, _ = make_coord()
    app_b = ServerApp(coord_b)
    httpd = serve(app_b, port=0)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        models_b, rec_b = _run_round_over(
            lambda app, cid: HttpTransport(url, cid, retries=1), coord_b)
    finally:
        httpd.shutdown()

    assert [r["new_version"] for r in rec_a] == [r["new_version"] for r in rec_b]
    for cid in CLIENTS:
        for k in models_a[cid]:
            np.testing.assert_allclose(models_a[cid][k], models_b[cid][k], atol=1e-12)


def test_unauthenticated_network_request_is_refused():
    coord, s = make_coord()
    httpd = serve(ServerApp(coord), port=0)
    try:
        import urllib.error
        import urllib.request
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/status"
        req = urllib.request.Request(url, data=encode(
            Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="status")),
            method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)      # no identity header at all
        assert e.value.code == 401
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------- client data boundary

def test_client_reads_only_its_own_shard(tmp_path):
    rng = np.random.default_rng(0)
    X, M, y = rng.standard_normal((40, 5)), np.ones((40, 5), np.int8), rng.standard_normal(40)
    folds = {"train": np.arange(24), "val": np.arange(24, 32), "test": np.arange(32, 40)}
    roles = {"always_observed": [0, 1], "maskable": [2, 3]}
    p0 = save_client_shard(str(tmp_path / "c0.npz"), X, M, y, folds, roles, "c0")
    rt = ClientRuntime(client_id="c1", experiment_id=EXP, shard_path=p0, transport=None)
    with pytest.raises(PermissionError):
        rt.load()                                        # c1 may not open c0's shard


def test_client_sends_only_aggregate_signature_and_count(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 4))
    M = (rng.random((60, 4)) > 0.3).astype(np.int8)
    y = rng.standard_normal(60)
    folds = {"train": np.arange(36), "val": np.arange(36, 48), "test": np.arange(48, 60)}
    roles = {"always_observed": [0], "maskable": [1, 2, 3]}
    p = save_client_shard(str(tmp_path / "c0.npz"), X, M, y, folds, roles, "c0")
    rt = ClientRuntime(client_id="c0", experiment_id=EXP, shard_path=p, transport=None,
                       bin_edges={"f0": list(np.linspace(-3, 3, 11))}, feature_names=[f"f{i}" for i in range(4)])
    rt.load()
    sig = rt.aggregate_signature()
    assert set(sig) == {"r", "H", "J", "C", "hist", "n_train"}
    assert sig["n_train"] == 36
    flat = json.dumps({k: np.asarray(v).tolist() for k, v in sig.items() if k != "hist"})
    assert str(round(float(X[0, 0]), 6)) not in flat      # no raw feature values leave


# ---------------------------------------------------------------- separate processes

@pytest.mark.slow
def test_independent_client_processes_complete_rounds(tmp_path):
    """1 server + 3 client PROCESSES complete a short multi-round run.
    Verifies the software boundary, not a scientific result."""
    import subprocess
    import sys
    import time

    shards = tmp_path / "shards"
    out = tmp_path / "out"
    env = {**os.environ, "PYTHONPATH": os.getcwd(), "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, "-m", "demo.prepare_shards", "--out", str(shards)],
                   check=True, capture_output=True, env=env, timeout=300)

    port_file = out / "port.txt"
    server = subprocess.Popen(
        [sys.executable, "-m", "demo.server", "--rounds", "2", "--port", "0",
         "--shards", str(shards), "--port-file", str(port_file),
         "--out", str(out / "summary.json"), "--linger", "8"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (port_file.exists() and port_file.read_text().strip()):
            if server.poll() is not None:
                pytest.fail(f"server exited early:\n{server.stdout.read()}")
            time.sleep(0.2)
        port = port_file.read_text().strip()
        assert port, "server never reported a port"

        procs = [subprocess.Popen(
            [sys.executable, "-m", "demo.client", "--client-id", c,
             "--url", f"http://127.0.0.1:{port}", "--shards", str(shards), "--rounds", "2"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for c in ["c0", "c1", "c2"]]
        logs = []
        for p in procs:
            logs.append(p.communicate(timeout=300)[0])
        for c, p, log in zip(["c0", "c1", "c2"], procs, logs):
            assert p.returncode == 0, f"{c} failed:\n{log}"
            assert "completed 2 rounds" in log
    finally:
        server.wait(timeout=60)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "timeout" not in summary and len(summary["rounds"]) == 2
    for rnd in summary["rounds"]:
        receivers = [r["receiver"] for r in rnd]
        assert sorted(receivers) == ["c0", "c1", "c2"]
        # personalised: a different aggregated model per receiver
        assert len({r["new_version"] for r in rnd}) == 3
        # declared sample counts are unequal, so weights must be non-uniform
        w = rnd[0]["weights"]
        assert len(set(np.round(list(w.values()), 6))) > 1


# ---------------------------------------------------- one contribution per client

def test_a_second_different_update_from_the_same_client_is_refused():
    """Regression: a different update from a client that had already reported
    was accepted and SILENTLY REPLACED its first contribution - letting one
    client choose which of its models is aggregated after watching the round."""
    coord, _ = make_coord()
    first, second = tiny_state(9), tiny_state(77)
    coord.submit(env_for(coord, "c0", first), first, "c0")
    kept = coord.round.updates["c0"]["w"].copy()

    with pytest.raises(UpdateRejected) as e:
        coord.submit(env_for(coord, "c0", second), second, "c0")
    assert e.value.reason is Reject.ALREADY_REPORTED

    np.testing.assert_array_equal(coord.round.updates["c0"]["w"], kept)
    assert len(coord.round.updates) == 1
    assert coord.missing() == [c for c in coord.clients if c != "c0"]


def test_an_identical_resend_is_still_a_duplicate_not_already_reported():
    """The two rejections are different facts and must stay distinguishable:
    a resend after an interruption is benign, a different model is not."""
    coord, _ = make_coord()
    s = tiny_state(9)
    coord.submit(env_for(coord, "c0", s), s, "c0")
    with pytest.raises(UpdateRejected) as e:
        coord.submit(env_for(coord, "c0", s), s, "c0")
    assert e.value.reason is Reject.DUPLICATE


def test_the_round_still_completes_after_a_refused_second_update():
    coord, _ = make_coord()
    for i, c in enumerate(coord.clients):
        s = tiny_state(2 + i)
        coord.submit(env_for(coord, c, s), s, c)
    other = tiny_state(99)
    with pytest.raises(UpdateRejected):
        coord.submit(env_for(coord, "c0", other), other, "c0")
    models, record = coord.close_round(
        lambda c, n: (_uniform(len(coord.clients), list(coord.clients).index(c)), 0.5))
    assert set(models) == set(coord.clients) and len(record) == len(coord.clients)


# ---------------------------------------------------- preprocessing agreement

def test_clients_declaring_different_preprocessing_coordinates_are_refused():
    """Averaging parameters fitted in different input coordinates is meaningless.

    Regression: every client fitted its own feature/target scale and the server
    averaged the results anyway.
    """
    coord, _ = make_coord()
    a, b = tiny_state(3), tiny_state(4)
    e_a = env_for(coord, "c0", a)
    e_a.meta["preprocessing_id"] = "shared:abc123"
    coord.submit(e_a, a, "c0")

    e_b = env_for(coord, "c1", b)
    e_b.meta["preprocessing_id"] = "per-client"
    with pytest.raises(UpdateRejected) as e:
        coord.submit(e_b, b, "c1")
    assert e.value.reason is Reject.PREPROCESSING
    assert "c1" not in coord.round.updates


def test_a_required_preprocessing_identity_is_enforced_on_every_update():
    coord, _ = make_coord()
    coord.require_preprocessing_id = "shared:deadbeef"
    s = tiny_state(3)
    e = env_for(coord, "c0", s)
    e.meta["preprocessing_id"] = "per-client"
    with pytest.raises(UpdateRejected) as err:
        coord.submit(e, s, "c0")
    assert err.value.reason is Reject.PREPROCESSING

    ok = env_for(coord, "c0", s)
    ok.meta["preprocessing_id"] = "shared:deadbeef"
    coord.submit(ok, s, "c0")
    assert "c0" in coord.round.updates


def test_shared_coordinates_put_every_client_in_the_same_input_space(tmp_path):
    """The positive case: given shared coordinates, two clients with very
    different local distributions standardise identically."""
    from mechfedgnn.client_runtime import (ClientRuntime, preprocessing_id, save_client_shard,
                                           shared_coordinates)
    rng = np.random.default_rng(0)
    shared = None
    transforms = {}
    for cid, shift, scale in (("c0", 0.0, 1.0), ("c1", 50.0, 9.0)):
        X = rng.standard_normal((40, 3)) * scale + shift
        M = np.ones((40, 3), np.int8)
        y = X[:, 0] * scale + shift
        folds = {"train": np.arange(24), "val": np.arange(24, 32), "test": np.arange(32, 40)}
        p = save_client_shard(str(tmp_path / f"{cid}.npz"), X, M, y, folds,
                              {"always_observed": [0], "maskable": [1, 2]}, cid)
        if shared is None:
            shared = shared_coordinates(X, M, y)
        rt = ClientRuntime(client_id=cid, experiment_id="demo", shard_path=p, transport=None,
                           shared_scale=shared)
        rt.load()
        transforms[cid] = rt.transform
        assert rt.preprocessing_id == preprocessing_id(shared)

    np.testing.assert_array_equal(transforms["c0"].mu, transforms["c1"].mu)
    np.testing.assert_array_equal(transforms["c0"].sd, transforms["c1"].sd)
    assert transforms["c0"].y_mu == transforms["c1"].y_mu

    # and without them, the clients are NOT in a common space - declared as such
    rt = ClientRuntime(client_id="c1", experiment_id="demo",
                       shard_path=str(tmp_path / "c1.npz"), transport=None)
    rt.load()
    assert rt.preprocessing_id == "per-client"
    assert not np.allclose(rt.transform.mu, transforms["c0"].mu)
