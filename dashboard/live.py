"""Live monitoring and controlled launching.

Two rules shape this file.

1. **Events are recorded, never invented.** Every entry in the event log is
   emitted at the moment something actually happened in the backend: a real
   request arriving at the coordinator, a real round opening, a real round
   closing with the weights it actually used. The dashboard animates a transfer
   only when an event of that kind appears. Nothing is synthesised to fill a gap
   or to make the picture livelier.

2. **Launching is frozen, not general.** There is no command field, no path
   field and no shell. The launcher can start exactly one thing - the bundled
   demonstration - with a handful of validated numeric options, each clamped to
   a small range. Client processes are started with a fixed argument vector.

The run itself is the real thing: the coordinator is the framework's
`ServerCoordinator` behind the framework's `ServerApp`, clients are separate OS
processes, and they speak over mutual TLS. Weighting goes through the research
code (`loop.pair_scores`, `loop.arm_weights`), not a copy of it.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from loop import SCORE_OF_ARM, arm_weights, pair_scores
from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.server import RoundTimeout, ServerCoordinator
from mechfedgnn.transport import ServerApp, serve

# Frozen launch envelope. These bounds are the authorisation model: there is
# nothing else the launcher can be asked to do.
LIMITS = {"clients": (2, 4), "rounds": (1, 3), "local_steps": (5, 50), "seed": (0, 9999)}
LIVE_ARMS = ["fedavg", "uniform-donor", "coverage-W_H", "coverage-marginal",
             "population-S", "combined-Q", "combined-Q-marginal", "marginal-rate",
             "missingness-similarity", "coverage-W_C"]
DEFAULTS = {"clients": 3, "rounds": 2, "local_steps": 25, "seed": 7, "arm": "combined-Q",
            "gamma": 0.5, "alpha": 1.0, "beta": 1.0, "lambda_pop": 0.5}


class EventBus:
    """Append-only, monotonically numbered, wall-clock stamped."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events = []

    def emit(self, kind, text, **fields):
        with self._lock:
            e = {"seq": len(self._events) + 1, "kind": kind, "text": text,
                 "t": time.time(), "mono": time.monotonic(), **fields}
            self._events.append(e)
            return e

    def since(self, seq):
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    def clear(self):
        with self._lock:
            self._events = []


class ObservedApp(ServerApp):
    """The framework's ServerApp, with an event emitted per REAL request."""

    def __init__(self, coordinator, bus, **kw):
        super().__init__(coordinator, **kw)
        self.bus = bus

    def handle(self, route, blob, authenticated_id):
        status, body = super().handle(route, blob, authenticated_id)
        if route == "/v1/status":
            return status, body                      # polling; not a transfer
        if route == "/v1/parent":
            self.bus.emit("transfer", f"server -> {authenticated_id}: parent model sent",
                          direction="out", client=authenticated_id, payload="parent_model",
                          status=status)
        elif route == "/v1/update":
            ok = status == 200
            self.bus.emit("transfer" if ok else "reject",
                          f"{authenticated_id} -> server: model update "
                          f"{'accepted' if ok else 'REJECTED'}"
                          + ("" if ok else f" ({json.loads(body.decode()).get('error')})"),
                          direction="in", client=authenticated_id, payload="model_update",
                          status=status)
        return status, body


def _si(sig: dict) -> dict:
    """An aggregate summary as the research scoring functions expect it."""
    return {"sig": {k: np.asarray(sig[k], float) for k in ("r", "H", "J", "C")},
            "hist": {k: np.asarray(v, float) for k, v in (sig.get("hist") or {}).items()},
            "n_train": sig.get("n_train")}


@dataclass
class Session:
    bus: EventBus
    state: str = "idle"          # idle | preparing | running | finished | failed
    cfg: dict = field(default_factory=dict)
    error: str = ""
    url: str = ""
    clients: list = field(default_factory=list)
    rounds_done: int = 0
    records: list = field(default_factory=list)       # real close_round output
    scores_seen: dict = field(default_factory=dict)   # round -> receiver -> donor -> score
    signature_clients: list = field(default_factory=list)
    _procs: list = field(default_factory=list)
    _httpd: object = None
    _thread: object = None
    _workdir: str = ""
    _stop: bool = False

    # ------------------------------------------------------------ lifecycle
    def start(self, cfg: dict):
        if self.state in ("preparing", "running"):
            raise RuntimeError("a run is already in progress")
        self.cfg = validate(cfg)
        self.bus.clear()
        self.state, self.error = "preparing", ""
        self.records, self.scores_seen, self.rounds_done = [], {}, 0
        self.signature_clients = []
        self._procs = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        if self._httpd is not None:
            threading.Thread(target=self._httpd.shutdown, daemon=True).start()
        self.bus.emit("control", "stop requested by the operator")

    def status(self, since=0):
        return {"state": self.state, "error": self.error, "cfg": self.cfg,
                "url": self.url, "clients": self.clients,
                "rounds_done": self.rounds_done, "rounds_total": self.cfg.get("rounds"),
                "records": self.records, "events": self.bus.since(since),
                "processes": [{"client": c, "pid": p.pid,
                               "running": p.poll() is None, "exit": p.poll()}
                              for c, p in zip(self.clients, self._procs)],
                "limits": LIMITS, "arms": LIVE_ARMS, "defaults": DEFAULTS}

    # ------------------------------------------------------------- the run
    def _run(self):
        work = tempfile.mkdtemp(prefix="mfg_live_")
        self._workdir = work
        shards, certs = os.path.join(work, "shards"), os.path.join(work, "certs")
        try:
            k, arm = self.cfg["clients"], self.cfg["arm"]
            self.clients = [f"c{i}" for i in range(k)]
            self.bus.emit("control", f"preparing {k} client shards (synthetic data, "
                                     f"seed {self.cfg['seed']})")
            _fixed_run([sys.executable, "-m", "demo.prepare_shards",
                        "--out", shards, "--clients", str(k), "--seed", str(self.cfg["seed"])])

            meta = json.load(open(os.path.join(shards, "meta.json"), encoding="utf-8"))
            from mechfedgnn.security import Allowlist, make_dev_certs, server_context
            make_dev_certs(certs, self.clients)
            self.bus.emit("control", "development certificates issued; mutual TLS required")
            self.bus.emit("control",
                          f"clients must declare preprocessing={meta.get('preprocessing_id')}; "
                          f"updates fitted in other coordinates are refused",
                          preprocessing_id=meta.get("preprocessing_id"))

            learner = MaskAwareMLP(hidden=tuple(meta["hidden"]))
            theta0 = learner.initial_state(meta["n_features"], seed=0)
            coord = ServerCoordinator(
                experiment_id="demo", clients=meta["clients"], schema=meta["schema_id"],
                round_timeout_s=120.0,
                # every client must standardise in the same coordinates, or its
                # parameters are not comparable with the others' and averaging
                # them means nothing
                require_preprocessing_id=meta.get("preprocessing_id"))
            app = ObservedApp(coord, self.bus,
                              allowlist=Allowlist({c: {"demo"} for c in self.clients}))
            self._httpd = serve(app, "127.0.0.1", 0, ssl_context=server_context(certs))
            port = self._httpd.server_address[1]
            self.url = f"https://127.0.0.1:{port}"     # serve() already started its thread
            self.bus.emit("control", f"coordinator listening on {self.url} (mutual TLS)",
                          url=self.url)

            needs_sig = SCORE_OF_ARM.get(arm) is not None
            for cid in self.clients:
                argv = [sys.executable, "-m", "demo.client", "--client-id", cid,
                        "--url", self.url, "--shards", shards,
                        "--rounds", str(self.cfg["rounds"]),
                        "--local-steps", str(self.cfg["local_steps"]), "--tls", certs]
                if needs_sig:
                    argv.append("--send-signature")
                self._procs.append(subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    cwd=os.getcwd()))
                self.bus.emit("process", f"{cid} started as a separate process "
                                         f"(pid {self._procs[-1].pid})", client=cid)

            self.state = "running"
            assignments = {c: theta0 for c in meta["clients"]}
            for r in range(1, self.cfg["rounds"] + 1):
                if self._stop:
                    break
                coord.open_round(r, assignments)
                self.bus.emit("round", f"round {r} open; waiting for {len(self.clients)} clients",
                              round=r)
                while coord.missing():
                    if self._stop:
                        raise RuntimeError("stopped by the operator")
                    self._reap()
                    coord.check_deadline()
                    time.sleep(0.05)
                if needs_sig:
                    self.signature_clients = sorted(coord.round.signatures)
                assignments, record = coord.close_round(self._weights_for(coord, arm, r))
                self.records.append(record)
                self.rounds_done = r
                for x in record:
                    self.bus.emit("aggregate",
                                  f"server -> {x['receiver']}: personalised model "
                                  f"(gamma {x['gamma']:.4f})",
                                  direction="out", client=x["receiver"], round=r,
                                  weights=x["weights"], gamma=x["gamma"])
                self.bus.emit("round", f"round {r} closed", round=r)
            self.state = "finished" if not self._stop else "idle"
            self.bus.emit("control", f"run {self.state}")
        except Exception as e:                                     # noqa: BLE001
            self.state = "failed"
            self.error = f"{type(e).__name__}: {e}"
            self.bus.emit("error", self.error)
        finally:
            self._drain()
            if self._httpd is not None:
                threading.Thread(target=self._httpd.shutdown, daemon=True).start()
            shutil.rmtree(work, ignore_errors=True)

    def _weights_for(self, coord, arm, round_id):
        """Real weighting, through the research code path.

        Falls back to the DECLARED fallback when a score is unusable - never to
        something invented here.
        """
        order = list(coord.clients)

        def weights_for(client, counts):
            sizes = np.array([counts[c] for c in order], float)
            p = sizes / sizes.sum()
            i = order.index(client)
            sigs = coord.round.signatures
            scores = [{} for _ in order]
            key = SCORE_OF_ARM.get(arm)
            if key and all(c in sigs for c in order):
                si = _si(sigs[client])
                for j, c in enumerate(order):
                    if j != i:
                        scores[j] = pair_scores(si, _si(sigs[c]), self.cfg["lambda_pop"])
                self.scores_seen.setdefault(round_id, {})[client] = {
                    order[j]: (None if scores[j].get(key) is None else float(scores[j][key]))
                    for j in range(len(order)) if j != i}
            elif key:
                self.bus.emit("fallback", f"{client}: signatures were not available from every "
                                          f"client, so `{arm}` cannot be scored this round",
                              client=client, round=round_id)
                scores = [{key: None} for _ in order]
            ac = {"alpha": self.cfg["alpha"], "beta": self.cfg["beta"],
                  "gamma": self.cfg["gamma"]}
            w, gamma, fb = arm_weights(arm, i, scores, p, ac)
            if fb:
                self.bus.emit("fallback", f"{client}: declared fallback ({fb}) - sample-size "
                                          f"weighting used", client=client, round=round_id,
                              reason=fb)
            return w, gamma

        return weights_for

    def _reap(self):
        for c, p in zip(self.clients, self._procs):
            if p.poll() not in (None, 0):
                raise RuntimeError(f"client {c} exited with code {p.poll()}")

    def _drain(self):
        for c, p in zip(self.clients, self._procs):
            try:
                if p.poll() is None:
                    out = p.communicate(timeout=8)[0]
                else:
                    out = p.stdout.read() if p.stdout else ""
            except subprocess.TimeoutExpired:
                p.kill()
                out = ""
            for line in (out or "").splitlines():
                self.bus.emit("client-log", f"{c}: {line}", client=c)


def _fixed_run(argv):
    """Run a fixed argument vector. No shell, no user-supplied program or path."""
    r = subprocess.run(argv, capture_output=True, text=True, cwd=os.getcwd())
    if r.returncode != 0:
        raise RuntimeError(f"{argv[2]} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def validate(cfg: dict) -> dict:
    """Clamp-and-reject. Anything not named here is dropped, not passed through."""
    out = dict(DEFAULTS)
    for k, (lo, hi) in LIMITS.items():
        if k in cfg:
            try:
                v = int(cfg[k])
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' must be a whole number")
            if not lo <= v <= hi:
                raise ValueError(f"'{k}' must be between {lo} and {hi}")
            out[k] = v
    if "arm" in cfg:
        if cfg["arm"] not in LIVE_ARMS:
            raise ValueError(f"'{cfg['arm']}' is not one of the permitted methods")
        out["arm"] = cfg["arm"]
    for k in ("gamma", "alpha", "beta", "lambda_pop"):
        if k in cfg:
            v = float(cfg[k])
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"'{k}' must be between 0 and 1")
            out[k] = v
    return out


# ------------------------------------------------------------------- routes

def routes(api, allow_launch=False):
    bus = EventBus()
    session = Session(bus=bus)

    def live_status(_api, q):
        since = int((q.get("since") or ["0"])[0])
        s = session.status(since)
        s["launch_enabled"] = allow_launch
        s["scores_seen"] = session.scores_seen
        s["signature_clients"] = session.signature_clients
        s["unavailable"] = {
            "per_step_training_loss": "clients report a model update, not a loss curve",
            "client_row_counts_beyond_n_train": "only the declared sample count is transmitted",
            "server_side_evaluation": "the server holds no data, so it cannot evaluate; "
                                      "RMSE for a live run would have to come from the clients "
                                      "and is not collected in this demonstration",
        }
        return s

    def live_start(_api, q, body):
        if not allow_launch:
            from dashboard.app import ApiError
            raise ApiError(403, "launching is disabled; start the dashboard with --allow-launch")
        session.start(body or {})
        return {"started": True, "cfg": session.cfg}

    def live_stop(_api, q, body):
        session.stop()
        return {"stopping": True}

    return {"live/status": live_status, "live/start": live_start, "live/stop": live_stop}
