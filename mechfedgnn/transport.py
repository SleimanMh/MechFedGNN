"""Transport: in-process and network execution through ONE logical protocol.

The same `ServerApp` serves both, so an in-process run and a networked run take
identical code paths through validation and aggregation - only the bytes' route
differs.

Authenticated identity:
  * in-process  - supplied directly by the harness;
  * network     - from the peer's TLS certificate when mutual TLS is on
                  (Milestone D), else from a development header that is
                  explicitly marked NOT a security control.

The client_id inside a message body is never trusted by itself; the server
compares it against the authenticated identity.
"""
import json
import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

import numpy as np

from mechfedgnn.protocol import (Envelope, MAX_PAYLOAD_BYTES, ProtocolError, decode, encode,
                                 state_version)
from mechfedgnn.server import Reject, RoundTimeout, ServerCoordinator, UpdateRejected

DEV_IDENTITY_HEADER = "X-Dev-Client-Id"   # development only; NOT authentication


class ServerApp:
    """Routes decoded messages to the coordinator. Transport-agnostic."""

    def __init__(self, coordinator: ServerCoordinator,
                 weights_for: Callable[[str, dict], tuple[np.ndarray, float]] | None = None,
                 max_bytes: int = MAX_PAYLOAD_BYTES):
        self.coord = coordinator
        self.weights_for = weights_for
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.rejections: list[dict] = []

    def handle(self, route: str, blob: bytes, authenticated_id: str) -> tuple[int, bytes]:
        """Returns (http-like status, response bytes). Never raises to the caller."""
        try:
            if len(blob) > self.max_bytes:
                raise ProtocolError("payload too large")
            env, state = decode(blob, self.max_bytes)
            with self.lock:
                if route == "/v1/parent":
                    if env.client_id != authenticated_id:
                        raise UpdateRejected(Reject.IDENTITY, "")
                    model, version = self.coord.parent_for(authenticated_id)
                    out = Envelope(experiment_id=self.coord.experiment_id,
                                   client_id=authenticated_id,
                                   round_id=self.coord.round.round_id,
                                   payload_type="parent_model", parent_version=version,
                                   schema_id=self.coord.schema)
                    return 200, encode(out, model)
                if route == "/v1/update":
                    uid = self.coord.submit(env, state, authenticated_id)
                    remaining = self.coord.missing()
                    return 200, self._ok({"accepted": uid, "missing": remaining})
                if route == "/v1/status":
                    r = self.coord.round
                    return 200, self._ok({"round": None if r is None else r.round_id,
                                          "closed": None if r is None else r.closed,
                                          "missing": [] if r is None else self.coord.missing()})
                raise ProtocolError(f"unknown route {route}")
        except UpdateRejected as e:
            self.rejections.append({"client": authenticated_id, "reason": e.reason.value})
            return 409, self._err(e.reason.value)
        except ProtocolError as e:
            self.rejections.append({"client": authenticated_id, "reason": Reject.MALFORMED.value})
            return 400, self._err(str(e))
        except RoundTimeout as e:
            return 408, self._err(str(e))
        except Exception:                       # never leak internals or payload contents
            self.rejections.append({"client": authenticated_id, "reason": "internal_error"})
            return 500, self._err("internal error")

    @staticmethod
    def _ok(d: Mapping[str, Any]) -> bytes:
        return json.dumps({"ok": True, **d}).encode()

    @staticmethod
    def _err(msg: str) -> bytes:
        return json.dumps({"ok": False, "error": msg}).encode()


# ---------------------------------------------------------------- in-process

@dataclass
class InProcessTransport:
    app: ServerApp
    client_id: str

    def call(self, route: str, env: Envelope, state=None) -> tuple[int, Any]:
        status, body = self.app.handle(route, encode(env, state), self.client_id)
        if route == "/v1/parent" and status == 200:
            return status, decode(body)
        return status, json.loads(body.decode())


# ---------------------------------------------------------------- network

def _make_handler(app: ServerApp, identity_from_cert: bool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "MechFedGNN/1.0"

        def log_message(self, fmt, *args):      # no request bodies in logs
            pass

        def _identity(self) -> str | None:
            if identity_from_cert:
                cert = self.connection.getpeercert() if hasattr(self.connection, "getpeercert") else None
                if not cert:
                    return None
                for field in cert.get("subject", ()):
                    for k, v in field:
                        if k == "commonName":
                            return v
                return None
            return self.headers.get(DEV_IDENTITY_HEADER)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > app.max_bytes:
                return self._respond(413, ServerApp._err("payload too large"))
            ident = self._identity()
            if not ident:
                return self._respond(401, ServerApp._err("unauthenticated"))
            blob = self.rfile.read(length) if length else b""
            status, body = app.handle(self.path, blob, ident)
            self._respond(status, body)

        def _respond(self, status: int, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(app: ServerApp, host: str = "127.0.0.1", port: int = 0,
          ssl_context: ssl.SSLContext | None = None) -> ThreadingHTTPServer:
    """Start a threaded server. With an mTLS context the peer certificate is the
    authenticated identity; without one, a development header is used and is
    explicitly NOT a security control."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(app, identity_from_cert=ssl_context is not None))
    if ssl_context is not None:
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@dataclass
class HttpTransport:
    url: str
    client_id: str
    ssl_context: ssl.SSLContext | None = None
    timeout_s: float = 30.0
    retries: int = 3
    backoff_s: float = 0.2

    def call(self, route: str, env: Envelope, state=None) -> tuple[int, Any]:
        blob = encode(env, state)
        headers = {"Content-Type": "application/octet-stream"}
        if self.ssl_context is None:
            headers[DEV_IDENTITY_HEADER] = self.client_id      # development only
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self.url + route, data=blob, headers=headers, method="POST")
            try:
                opener = (urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=self.ssl_context))
                    if self.ssl_context else urllib.request.build_opener())
                with opener.open(req, timeout=self.timeout_s) as r:
                    body = r.read()
                    if route == "/v1/parent":
                        return r.status, decode(body)
                    return r.status, json.loads(body.decode())
            except urllib.error.HTTPError as e:                 # a definite answer: do not retry
                body = e.read()
                try:
                    return e.code, json.loads(body.decode())
                except Exception:
                    return e.code, {"ok": False, "error": "unreadable error body"}
            except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionError) as e:
                last = e
                import time as _t
                _t.sleep(self.backoff_s * (2 ** attempt))       # retry transient faults only
        raise ConnectionError(f"{route}: no response after {self.retries} attempts ({last})")
