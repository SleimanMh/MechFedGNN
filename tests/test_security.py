"""Milestone D: transport security.

Verifies the trust boundary, not production readiness. See docs/SECURITY.md for
what this does and does NOT protect against.
"""
import json
import os
import ssl

import numpy as np
import pytest

from mechfedgnn.protocol import Envelope, encode, schema_id
from mechfedgnn.security import Allowlist, client_context, make_dev_certs, server_context
from mechfedgnn.server import Reject, ServerCoordinator
from mechfedgnn.transport import HttpTransport, ServerApp, serve

CLIENTS = ["c0", "c1", "c2"]
EXP = "demo"


def tiny_state(scale=1.0):
    return {"w": np.arange(4.0) * scale, "b": np.array([0.5 * scale])}


def fresh_coord():
    coord = ServerCoordinator(experiment_id=EXP, clients=CLIENTS, schema=schema_id(tiny_state()))
    coord.open_round(1, {c: tiny_state(1 + i) for i, c in enumerate(CLIENTS)})
    return coord


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    d = tmp_path_factory.mktemp("certs")
    make_dev_certs(str(d), CLIENTS)
    return str(d)


@pytest.fixture
def tls_server(certs):
    coord = fresh_coord()
    app = ServerApp(coord, allowlist=Allowlist({c: [EXP] for c in CLIENTS}))
    httpd = serve(app, "127.0.0.1", 0, ssl_context=server_context(certs))
    yield f"https://127.0.0.1:{httpd.server_address[1]}", app, coord
    httpd.shutdown()


def test_mutual_tls_round_trip_uses_certificate_identity(tls_server, certs):
    url, _app, coord = tls_server
    t = HttpTransport(url, "c0", ssl_context=client_context(certs, "c0"), retries=1)
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="parent_request")
    status, (penv, _state) = t.call("/v1/parent", env)
    assert status == 200 and penv.parent_version == coord.round.assigned["c0"]


def test_unknown_client_certificate_is_rejected(tls_server, tmp_path):
    """A certificate minted by a DIFFERENT CA must not complete the handshake."""
    url, _app, _coord = tls_server
    rogue = str(tmp_path / "rogue")
    make_dev_certs(rogue, ["c0"])
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(os.path.join(rogue, "ca.pem"))
    ctx.load_cert_chain(os.path.join(rogue, "client_c0.pem"), os.path.join(rogue, "client_c0.key"))
    t = HttpTransport(url, "c0", ssl_context=ctx, retries=1)
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="parent_request")
    with pytest.raises((ConnectionError, ssl.SSLError, OSError)):
        t.call("/v1/parent", env)


def test_incorrect_server_certificate_causes_connection_failure(tls_server, tmp_path, certs):
    """The client trusts a different CA, so server verification must fail."""
    url, _app, _coord = tls_server
    other = str(tmp_path / "other")
    make_dev_certs(other, ["c0"])
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(os.path.join(other, "ca.pem"))
    ctx.load_cert_chain(os.path.join(certs, "client_c0.pem"),
                        os.path.join(certs, "client_c0.key"))
    t = HttpTransport(url, "c0", ssl_context=ctx, retries=1)
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="parent_request")
    with pytest.raises((ConnectionError, ssl.SSLError, OSError)):
        t.call("/v1/parent", env)


def test_certificate_and_request_identity_mismatch_is_rejected(tls_server, certs):
    """Authenticated by certificate as c1, but the body claims to be c0."""
    url, _app, coord = tls_server
    t = HttpTransport(url, "c1", ssl_context=client_context(certs, "c1"), retries=1)
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="model_update",
                   parent_version=coord.round.assigned["c0"], schema_id=coord.schema,
                   update_id="x", meta={"n_train": 10})
    status, resp = t.call("/v1/update", env, tiny_state())
    assert status == 409 and resp["error"] == Reject.IDENTITY.value


def test_authenticated_identity_not_enrolled_in_this_experiment_is_refused():
    app = ServerApp(fresh_coord(), allowlist=Allowlist({"c0": ["a-different-experiment"]}))
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="status")
    status, body = app.handle("/v1/status", encode(env), "c0")
    assert status == 403 and "not enrolled" in json.loads(body)["error"]


def test_identity_absent_from_the_allowlist_is_refused():
    app = ServerApp(fresh_coord(), allowlist=Allowlist({"c0": [EXP]}))
    env = Envelope(experiment_id=EXP, client_id="c9", round_id=1, payload_type="status")
    status, body = app.handle("/v1/status", encode(env), "c9")
    assert status == 403 and "unknown client certificate identity" in json.loads(body)["error"]


def test_oversized_malformed_and_non_finite_payloads_are_rejected_before_aggregation():
    coord = fresh_coord()
    app = ServerApp(coord, max_bytes=2048)
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=1, payload_type="model_update",
                   parent_version=coord.round.assigned["c0"], schema_id=coord.schema,
                   update_id="u", meta={"n_train": 5})

    status, body = app.handle("/v1/update", encode(env, {"w": np.zeros(100000),
                                                         "b": np.zeros(1)}), "c0")
    assert status == 400 and "too large" in json.loads(body)["error"]

    status, _ = app.handle("/v1/update", b"not-a-message", "c0")
    assert status == 400

    bad = {"w": np.array([1.0, np.nan, 2.0, 3.0]), "b": np.array([0.0])}
    status, body = app.handle("/v1/update", encode(env, bad), "c0")
    assert status == 400 and "non-finite" in json.loads(body)["error"]

    assert coord.round.updates == {}          # nothing reached aggregation


def test_development_certificates_are_marked_unsuitable_for_production(certs):
    from cryptography import x509
    with open(os.path.join(certs, "client_c0.pem"), "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    org = cert.subject.get_attributes_for_oid(x509.oid.NameOID.ORGANIZATION_NAME)[0].value
    assert "DO-NOT-USE-IN-PRODUCTION" in org
    assert os.path.exists(os.path.join(certs, "README.txt"))


def test_logs_and_error_bodies_carry_no_payload_values_or_key_material(tls_server, certs, capsys):
    url, app, coord = tls_server
    t = HttpTransport(url, "c0", ssl_context=client_context(certs, "c0"), retries=1)
    secret = 424242.125
    state = {"w": np.array([secret, 1.0, 2.0, 3.0]), "b": np.array([0.0])}
    env = Envelope(experiment_id=EXP, client_id="c0", round_id=99, payload_type="model_update",
                   parent_version="wrong", schema_id=coord.schema, update_id="u",
                   meta={"n_train": 5})
    status, resp = t.call("/v1/update", env, state)
    assert status == 409
    blob = json.dumps(resp) + json.dumps(app.rejections) + capsys.readouterr().out
    assert str(secret) not in blob
    for marker in ["BEGIN PRIVATE KEY", "BEGIN EC PRIVATE KEY"]:
        assert marker not in blob


def test_no_credential_like_files_are_tracked_by_git():
    import subprocess
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    tracked = [l for l in out.splitlines() if l.endswith((".key", ".pem", ".crt", ".p12", ".pfx"))]
    assert tracked == [], f"credential-like files are tracked: {tracked}"
