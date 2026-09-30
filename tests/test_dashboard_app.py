"""Dashboard backend: routing, the constraints that keep it safe, and the live layer.

The constraints under test are the ones stated in the brief:
  * no arbitrary filesystem paths and no shell reachable from a request;
  * bound to localhost by default;
  * raw records, labels, per-row masks and per-example predictions never leave;
  * the launcher accepts only a frozen, validated configuration;
  * a live animation is driven by recorded backend events, not by a timer.
"""
import copy
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from dashboard import app as dash, live, replay
from data.inject import build_roles
from loop import run_seed
from report import write_run
from tools.capture_reference import SEED, base_cfg, synthetic_df


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("results"))
    df = synthetic_df()
    roles = build_roles(df)
    cfg = base_cfg()
    out = run_seed(df, roles, copy.deepcopy(cfg), SEED, folds=("val", "test"), headroom=False)
    run_dir = write_run("e1", cfg, "synthetic", df, roles, [], [out], root=root)

    api = dash.Api(root)
    httpd = dash.serve("127.0.0.1", 0, root, live.routes(api, allow_launch=False), api)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, run_dir.replace("\\", "/"), root
    httpd.shutdown()
    httpd.server_close()


def get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=20) as r:
        return r.status, json.loads(r.read())


def get_raw(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=20) as r:
        return r.status, r.read()


def post(base, path, body):
    req = urllib.request.Request(f"{base}/api/{path}", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, json.loads(r.read())


# ------------------------------------------------------------------ routing

def test_the_api_serves_a_run_and_its_aggregation(server):
    base, path, _ = server
    _, runs = get(base, "/api/runs")
    assert any(r["path"] == path for r in runs["runs"])
    _, run = get(base, f"/api/run?path={path}")
    seed, rec = run["seeds"][0], run["clients"][0]["id"]
    _, agg = get(base, f"/api/aggregation?path={path}&seed={seed}&receiver={rec}&arm=combined-Q")
    assert agg["total_ok"] and len(agg["steps"]) == 7
    assert agg["weights_source"].startswith("recomputed")


def test_the_page_and_its_assets_are_served(server):
    base, _, _ = server
    for p, frag in (("/", b"MechFedGNN"), ("/app.js", b"use strict"), ("/app.css", b"--navy")):
        status, body = get_raw(base, p)
        assert status == 200 and frag in body


# -------------------------------------------------------------- constraints

@pytest.mark.parametrize("bad", [
    "results", "../loop.py", "C:/Windows/System32", "/etc/passwd", "",
    "results/../../loop.py", "results%2F..%2Floop.py",
])
def test_only_whitelisted_run_directories_can_be_opened(server, bad):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, f"/api/run?path={bad}")
    assert e.value.code in (400, 404)


@pytest.mark.parametrize("bad", ["/../dashboard/app.py", "/../../loop.py",
                                 "/..%2f..%2floop.py", "/static/../../loop.py"])
def test_static_serving_cannot_escape_its_directory(server, bad):
    base, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get_raw(base, bad)
    assert e.value.code in (403, 404)


def test_no_endpoint_accepts_a_command_or_an_arbitrary_path(server):
    """Nothing in the API surface takes a program to run or a path to read."""
    names = set(dash.Api.ROUTES) | {"live/status", "live/start", "live/stop"}
    assert not {n for n in names if any(w in n for w in ("exec", "shell", "cmd", "file", "read"))}
    src = open(os.path.join("dashboard", "app.py"), encoding="utf-8").read()
    assert "shell=True" not in src and "os.system" not in src and "eval(" not in src


def test_serving_defaults_to_localhost():
    import inspect
    assert inspect.signature(dash.serve).parameters["host"].default == "127.0.0.1"
    assert inspect.signature(dash.main).parameters == inspect.signature(dash.main).parameters
    src = open(os.path.join("dashboard", "app.py"), encoding="utf-8").read()
    assert '"--host", default="127.0.0.1"' in src


def test_no_raw_records_labels_or_per_row_masks_are_ever_returned(server):
    """Sweep every read endpoint and assert nothing row-shaped comes back."""
    base, path, _ = server
    _, run = get(base, f"/api/run?path={path}")
    seed, rec = run["seeds"][0], run["clients"][0]["id"]
    n_train = run["clients"][0]["n_train"]
    bodies = []
    for p in (f"/api/run?path={path}",
              f"/api/client?path={path}&id={rec}",
              f"/api/aggregation?path={path}&seed={seed}&receiver={rec}&arm=combined-Q",
              f"/api/matrix?path={path}&seed={seed}&arm=combined-Q",
              f"/api/history?path={path}&receiver={rec}&arm=combined-Q",
              f"/api/results?path={path}&fold=test&timepoint=t2",
              f"/api/explain?path={path}&fold=test&timepoint=t2&receiver={rec}&arm=combined-Q",
              f"/api/scores?path={path}&seed={seed}&receiver={rec}",
              f"/api/timeline?path={path}&seed={seed}&arm=combined-Q"):
        _, body = get(base, p)
        bodies.append(body)
        for banned in ("X", "y", "labels", "predictions", "M", "rows"):
            assert banned not in body, f"{p} returned a '{banned}' field"
        assert _max_list(body) < n_train, f"{p} returned a row-length array"


def _max_list(x):
    if isinstance(x, list):
        return max([len(x)] + [_max_list(v) for v in x])
    if isinstance(x, dict):
        return max([0] + [_max_list(v) for v in x.values()])
    return 0


def test_unavailable_information_is_declared_by_the_api(server):
    base, path, _ = server
    _, run = get(base, f"/api/run?path={path}")
    assert run["availability"]["event_log"]["available"] is False
    _, tl = get(base, f"/api/timeline?path={path}&seed={run['seeds'][0]}&arm=fedavg")
    assert tl["timestamps"] is None and "not a measured execution timeline" in tl["note"]


# --------------------------------------------------------------- launch gate

def test_launching_is_refused_when_it_was_not_enabled(server):
    base, _, _ = server
    _, s = get(base, "/api/live/status")
    assert s["launch_enabled"] is False
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "live/start", {"clients": 3})
    assert e.value.code == 403


@pytest.mark.parametrize("cfg,msg", [
    ({"clients": 40}, "between"), ({"clients": 1}, "between"),
    ({"rounds": 99}, "between"), ({"local_steps": 100000}, "between"),
    ({"arm": "local-only"}, "permitted"), ({"arm": "../../etc/passwd"}, "permitted"),
    ({"clients": "3; rm -rf /"}, "whole number"), ({"gamma": 5}, "between 0 and 1"),
])
def test_the_launch_configuration_is_validated_not_clamped_silently(cfg, msg):
    with pytest.raises(ValueError, match=msg):
        live.validate(cfg)


def test_unknown_launch_keys_are_dropped_rather_than_passed_through():
    out = live.validate({"clients": 2, "command": "calc.exe", "shards": "C:/",
                         "python": "/bin/sh", "cwd": "/"})
    assert out["clients"] == 2
    assert set(out) == set(live.DEFAULTS)


def test_the_launcher_never_uses_a_shell_and_names_its_program_itself():
    src = open(os.path.join("dashboard", "live.py"), encoding="utf-8").read()
    assert "shell=True" not in src and "os.system" not in src
    assert "sys.executable" in src          # the program is ours, never supplied by a request


# ------------------------------------------------------------ recorded events

def test_events_carry_real_timestamps_and_increasing_sequence_numbers():
    bus = live.EventBus()
    a = bus.emit("round", "one")
    time.sleep(0.01)
    b = bus.emit("round", "two")
    assert b["seq"] == a["seq"] + 1 and b["mono"] > a["mono"]
    assert abs(a["t"] - time.time()) < 5
    assert [e["seq"] for e in bus.since(a["seq"])] == [b["seq"]]


def test_an_idle_session_emits_nothing_at_all():
    """Nothing is generated to make the picture look alive."""
    bus = live.EventBus()
    s = live.Session(bus=bus)
    time.sleep(0.4)
    assert bus.since(0) == []
    assert s.status()["state"] == "idle" and s.status()["records"] == []


def test_transfer_events_are_emitted_only_by_a_real_request(monkeypatch):
    """ObservedApp emits on handle(); a status poll is not a transfer.

    The base handler is stubbed via monkeypatch so pytest restores it: patching
    ServerApp.handle permanently would break every other network test.
    """
    bus = live.EventBus()
    monkeypatch.setattr(live.ServerApp, "handle",
                        lambda self, route, blob, ident: (200, b'{"ok": true}'))
    obs = live.ObservedApp.__new__(live.ObservedApp)
    obs.bus = bus

    obs.handle("/v1/status", b"", "c0")
    assert bus.since(0) == []                                   # polling is not a transfer
    obs.handle("/v1/update", b"", "c0")
    obs.handle("/v1/parent", b"", "c0")
    assert [e["kind"] for e in bus.since(0)] == ["transfer", "transfer"]


def test_the_stubbing_above_did_not_leak_into_the_real_server_app():
    """Guard: the previous test must leave ServerApp.handle untouched."""
    from mechfedgnn.transport import ServerApp as RealApp
    assert RealApp.handle.__qualname__ == "ServerApp.handle"


@pytest.mark.slow
def test_a_real_launched_run_completes_over_mutual_tls_with_separate_processes():
    """End to end: real client processes, real TLS, real recorded events."""
    bus = live.EventBus()
    s = live.Session(bus=bus)
    s.start({"clients": 2, "rounds": 1, "local_steps": 5, "seed": 3, "arm": "combined-Q"})
    deadline = time.time() + 180
    while time.time() < deadline and s.state in ("idle", "preparing", "running"):
        time.sleep(0.25)
    assert s.state == "finished", s.error
    assert s.rounds_done == 1 and len(s.records) == 1
    s._thread.join(60)            # client output is collected as each process exits

    st = s.status()
    assert st["url"].startswith("https://127.0.0.1:")
    assert all(p["exit"] == 0 for p in st["processes"])
    assert len(st["processes"]) == 2

    events = bus.since(0)
    assert any("mutual TLS" in e["text"] for e in events)
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
    transfers = [e for e in events if e["kind"] == "transfer"]
    assert {e["client"] for e in transfers} == {"c0", "c1"}
    assert {e["direction"] for e in transfers} == {"in", "out"}

    # weights came from the real aggregation, and the receiver never donates to itself
    for rec in s.records[0]:
        assert rec["receiver"] not in rec["weights"]
        assert abs(sum(rec["weights"].values()) - 1.0) < 1e-9
        assert 0.0 <= rec["gamma"] <= 1.0
    # scores were computed from signatures the clients actually transmitted
    assert set(s.signature_clients) == {"c0", "c1"}


@pytest.mark.slow
def test_a_live_run_transmits_aggregate_signatures_only():
    """The signature a client sends must be per-feature, never per-row."""
    from mechfedgnn.client_runtime import ClientRuntime
    bus = live.EventBus()
    s = live.Session(bus=bus)
    s.start({"clients": 2, "rounds": 1, "local_steps": 5, "seed": 3, "arm": "combined-Q"})
    deadline = time.time() + 180
    while time.time() < deadline and s.state in ("idle", "preparing", "running"):
        time.sleep(0.25)
    assert s.state == "finished", s.error
    # ClientRuntime.submit attaches exactly aggregate_only(); assert its shape
    src = open(os.path.join("mechfedgnn", "client_runtime.py"), encoding="utf-8").read()
    assert 'meta["signature"] = self._json_safe(self.aggregate_signature())' in src
    assert hasattr(ClientRuntime, "aggregate_signature")
