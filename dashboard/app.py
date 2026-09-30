"""Dashboard backend: read-only HTTP over the framework.

Design constraints this file enforces:

  * binds 127.0.0.1 by default; --host must be given explicitly to change that;
  * NO arbitrary filesystem access - a `path` parameter is only accepted if it
    is in the whitelist produced by `replay.list_runs()`;
  * NO shell execution of any kind reachable from a request;
  * responses carry aggregate quantities only. Raw client records, labels,
    per-row masks and per-example predictions are never assembled here, so they
    cannot be sent by accident.

All training, scoring, weighting and statistics are computed by the framework
and by `dashboard.replay`; this module only routes and serialises.
"""
import argparse
import json
import mimetypes
import os
import posixpath
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from dashboard import replay

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 64 * 1024


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if v != v or v in (float("inf"), float("-inf")) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Api:
    """Route handlers. Held in one object so the live layer can extend it."""

    def __init__(self, results_root=replay.RESULTS_ROOT):
        self.results_root = results_root
        self._lock = threading.Lock()
        self._allowed = {}
        self.refresh_runs()

    def refresh_runs(self):
        with self._lock:
            runs = replay.list_runs(self.results_root)
            self._allowed = {r["path"]: r for r in runs}
            return runs

    def _run(self, q):
        """Resolve `path` against the whitelist. Anything else is refused."""
        path = (q.get("path") or [""])[0]
        with self._lock:
            known = path in self._allowed
        if not known:
            self.refresh_runs()
            with self._lock:
                known = path in self._allowed
        if not known:
            raise ApiError(404, "unknown run; only completed runs under the results "
                                "directory can be opened")
        return replay.load_run(path)

    @staticmethod
    def _need(q, key):
        v = (q.get(key) or [None])[0]
        if v is None:
            raise ApiError(400, f"missing parameter '{key}'")
        return v

    # ---- routes ----------------------------------------------------------
    def runs(self, q):
        return {"runs": self.refresh_runs(), "results_root": self.results_root}

    def run(self, q):
        r = self._run(q)
        return {k: r[k] for k in ("run_id", "path", "experiment", "dataset", "method",
                                  "seeds", "clients", "availability")} | {
            "axes": replay.folds_and_timepoints(r),
            "config_summary": _config_summary(r["config"]),
        }

    def client(self, q):
        r = self._run(q)
        cid = self._need(q, "id")
        c = next((c for c in r["clients"] if c["id"] == cid), None)
        if c is None:
            raise ApiError(404, f"no client '{cid}' in this run")
        p = r["params"].get(cid, {})
        feats = c["feature_names"]
        rates = list(p.get("r") or [])
        return {**c,
                "observed_rate": [1.0 - x for x in rates],
                "missing_rate": rates,
                "feature_rows": [{"feature": feats[i] if i < len(feats) else f"f{i}",
                                  "missing_rate": rates[i],
                                  "maskable": (feats[i] if i < len(feats) else None) in c["maskable"]}
                                 for i in range(len(rates))],
                "histogram_keys": sorted((p.get("hist") or {}).keys()),
                "shared_counts": {"J_available": "J" in p, "H_available": "H" in p,
                                  "C_available": "C" in p},
                "privacy": "these are aggregate mask summaries and coarse histograms - "
                           "the rows, labels and per-row masks never leave the client"}

    def aggregation(self, q):
        r = self._run(q)
        return replay.aggregation_breakdown(r, int(self._need(q, "seed")),
                                            self._need(q, "receiver"), self._need(q, "arm"))

    def matrix(self, q):
        r = self._run(q)
        return replay.weight_matrix(r, int(self._need(q, "seed")), self._need(q, "arm"))

    def history(self, q):
        r = self._run(q)
        return replay.weight_history(r, self._need(q, "receiver"), self._need(q, "arm"))

    def results(self, q):
        r = self._run(q)
        fold = (q.get("fold") or ["test"])[0]
        tp = (q.get("timepoint") or ["t2"])[0]
        ref = (q.get("reference") or ["uniform-donor"])[0]
        metric = (q.get("metric") or ["rmse"])[0]
        return {"per_receiver": replay.per_receiver_metric(r, fold, tp, metric),
                "contrasts": replay.contrasts(r, fold, tp, ref)}

    def explain(self, q):
        r = self._run(q)
        return replay.result_explanation(r, (q.get("fold") or ["test"])[0],
                                         (q.get("timepoint") or ["t2"])[0],
                                         self._need(q, "receiver"), self._need(q, "arm"))

    def scores(self, q):
        r = self._run(q)
        return {"receiver": self._need(q, "receiver"),
                "scores": replay.score_table(r, int(self._need(q, "seed")),
                                             self._need(q, "receiver"))}

    def timeline(self, q):
        """A research run's snapshot 'timeline'. Explicitly NOT an execution trace."""
        r = self._run(q)
        seed = int(self._need(q, "seed"))
        arm = self._need(q, "arm")
        stages = [
            {"id": "partition", "label": "Partition and inject missingness",
             "detail": f"{len(r['clients'])} clients built from {r['dataset']}, seed {seed}"},
            {"id": "local", "label": "Local training",
             "detail": f"each client trains on its own rows for "
                       f"{r['method'].get('local_steps')} steps"},
            {"id": "signature", "label": "Signature upload",
             "detail": "aggregate mask summaries and histograms only"},
            {"id": "scoring", "label": "Scoring",
             "detail": f"arm `{arm}` scores every receiver-donor pair"},
            {"id": "weighting", "label": "Weighting", "detail": "scores become donor weights"},
            {"id": "aggregation", "label": "Aggregation",
             "detail": "one personalised model per receiver"},
            {"id": "adapt", "label": "Local adaptation",
             "detail": f"{r['method'].get('adapt_budget')} adaptation steps (timepoint t2)"},
            {"id": "evaluate", "label": "Evaluation", "detail": "RMSE on held-out rows"},
        ]
        return {"stages": stages, "ordered": True, "timestamps": None,
                "note": "this run recorded no timestamps or events, so the order below is the "
                        "protocol's fixed order - not a measured execution timeline",
                "rounds": 1,
                "rounds_note": "a research run is a single federated round; `budget` values are "
                               "local adaptation steps, not rounds"}

    ROUTES = {"runs": runs, "run": run, "client": client, "aggregation": aggregation,
              "matrix": matrix, "history": history, "results": results, "explain": explain,
              "scores": scores, "timeline": timeline}


def _config_summary(cfg):
    ac = cfg.get("aggregation", {})
    md = cfg.get("model", {})
    return {"aggregation": ac, "model": md,
            "injection": {k: cfg.get(k) for k in ("mechanism", "panels", "clients", "seeds")
                          if k in cfg},
            "experiment_markers": {k: cfg[k] for k in ("e5", "e1m", "corrected", "assignment")
                                   if k in cfg}}


class Handler(BaseHTTPRequestHandler):
    server_version = "MechFedGNN-Dashboard"
    api: Api = None
    extra_routes: dict = {}

    def log_message(self, fmt, *args):
        # Query strings can name clients; keep them out of the log entirely.
        path = urlparse(self.path).path
        print(f"[dashboard] {self.command} {path} {args[1] if len(args) > 1 else ''}")

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/"):
            return self._api(u.path[len("/api/"):], parse_qs(u.query), None)
        return self._static(u.path)

    def do_POST(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._send(413, {"error": "request body too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "malformed JSON body"})
        return self._api(u.path[len("/api/"):], parse_qs(u.query), body)

    def _api(self, name, q, body):
        route = self.extra_routes.get(name) or Api.ROUTES.get(name)
        if route is None:
            return self._send(404, {"error": f"no endpoint '{name}'"})
        try:
            out = route(self.api, q) if body is None else route(self.api, q, body)
        except ApiError as e:
            return self._send(e.status, {"error": e.message})
        except (KeyError, ValueError) as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:                                    # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(200, jsonable(out))

    def _static(self, path):
        """Serve the frontend only. The static root is the only readable directory."""
        rel = posixpath.normpath(path.lstrip("/") or "index.html")
        if rel in (".", "..") or rel.startswith("../") or os.path.isabs(rel):
            return self._send(403, {"error": "forbidden"})
        full = os.path.normpath(os.path.join(STATIC, *rel.split("/")))
        if os.path.commonpath([os.path.abspath(full), STATIC]) != STATIC:
            return self._send(403, {"error": "forbidden"})
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            return self._send(200, f.read(), ctype)


def build(results_root=replay.RESULTS_ROOT, extra_routes=None, api=None):
    handler = type("BoundHandler", (Handler,),
                   {"api": api or Api(results_root), "extra_routes": extra_routes or {}})
    return handler


def serve(host="127.0.0.1", port=8765, results_root=replay.RESULTS_ROOT, extra_routes=None,
          api=None):
    httpd = ThreadingHTTPServer((host, port), build(results_root, extra_routes, api))
    return httpd


def main():
    ap = argparse.ArgumentParser(description="MechFedGNN experiment dashboard (read-only)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; localhost by default and deliberately so")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results", default=replay.RESULTS_ROOT)
    ap.add_argument("--allow-launch", action="store_true",
                    help="enable the launch controls (small, frozen configurations only)")
    args = ap.parse_args()

    extra = {}
    api = Api(args.results)
    try:
        from dashboard import live
        extra.update(live.routes(api, allow_launch=args.allow_launch))
    except ImportError:
        print("[dashboard] live monitoring module not present; replay only")

    httpd = serve(args.host, args.port, args.results, extra, api)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[dashboard] WARNING: binding {args.host} exposes this dashboard beyond this "
              f"machine. It has no authentication.")
    print(f"[dashboard] http://{args.host}:{args.port}  (results: {args.results}, "
          f"launch {'enabled' if args.allow_launch else 'disabled'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
