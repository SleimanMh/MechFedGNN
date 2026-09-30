"""Demonstration client: one independent process, one data shard.

Reads ONLY its own shard. Sends a model update and its declared sample count.
Never sends features, labels, per-row masks or per-example predictions.

Run:  python -m demo.client --client-id c0 --url http://127.0.0.1:8443 --rounds 3
"""
import argparse
import json
import os
import time

from mechfedgnn.client_runtime import ClientRuntime
from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.protocol import Envelope
from mechfedgnn.transport import HttpTransport


def _status(transport, client_id, round_id):
    env = Envelope(experiment_id="demo", client_id=client_id, round_id=round_id,
                   payload_type="status")
    status, resp = transport.call("/v1/status", env)
    return resp if status == 200 else {}


def wait_open(transport, client_id, r, timeout_s):
    """Wait until round r is the open round. If the server has already moved
    past r, this client missed it - report rather than hang."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = _status(transport, client_id, r)
        cur = s.get("round")
        if cur == r and not s.get("closed"):
            return True
        if cur is not None and cur > r:
            return False
        time.sleep(0.05)
    return False


def wait_done(transport, client_id, r, timeout_s):
    """Wait until round r is finished. The server opens r+1 immediately after
    closing r, so 'round has advanced past r' also means r is done - polling for
    the closed flag alone would race and never match."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = _status(transport, client_id, r)
        cur = s.get("round")
        if cur is not None and (cur > r or (cur == r and s.get("closed"))):
            return True
        time.sleep(0.05)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--shards", default=os.path.join("demo", "_shards"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local-steps", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--tls", default=None, help="certificate directory for mutual TLS")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.shards, "meta.json"), encoding="utf-8"))
    ctx = None
    if args.tls:
        from mechfedgnn.security import client_context
        ctx = client_context(args.tls, args.client_id)
    transport = HttpTransport(args.url, args.client_id, ssl_context=ctx, timeout_s=args.timeout)

    rt = ClientRuntime(client_id=args.client_id, experiment_id="demo",
                       shard_path=os.path.join(args.shards, f"{args.client_id}.npz"),
                       transport=transport, learner=MaskAwareMLP(hidden=tuple(meta["hidden"])),
                       local_steps=args.local_steps, seed=1000,
                       bin_edges=meta["bin_edges"], feature_names=meta["feature_names"])
    rt.load()
    print(f"{args.client_id}: shard loaded, n_train={rt.n_train}", flush=True)

    for r in range(1, args.rounds + 1):
        if not wait_open(transport, args.client_id, r, args.timeout):
            print(f"{args.client_id}: round {r} never opened", flush=True)
            raise SystemExit(2)
        status, resp = rt.run_round(r, meta["schema_id"])
        print(f"{args.client_id}: round {r} submitted -> {status} {resp}", flush=True)
        if status != 200:
            raise SystemExit(3)
        if not wait_done(transport, args.client_id, r, args.timeout):
            print(f"{args.client_id}: round {r} never closed", flush=True)
            raise SystemExit(4)
    print(f"{args.client_id}: completed {args.rounds} rounds", flush=True)


if __name__ == "__main__":
    main()
