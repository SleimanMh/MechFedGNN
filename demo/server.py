"""Demonstration server: coordinates a short multi-round federated run.

Holds NO client data. It sees model updates, declared sample counts and (when
enabled) aggregate signatures - nothing else.

This demonstration verifies the framework. It is NOT a new scientific result,
and the historical single-round research protocol is unchanged.

Run:  python -m demo.server --rounds 3 --port 8443 [--tls demo/_certs]
"""
import argparse
import json
import os
import time

import numpy as np

from mechfedgnn.learner import MaskAwareMLP
from mechfedgnn.server import RoundTimeout, ServerCoordinator
from mechfedgnn.transport import ServerApp, serve
from mechfedgnn.weighting import policy_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join("demo", "_shards"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--on-timeout", default="fail", choices=["fail", "pause"])
    ap.add_argument("--tls", default=None, help="certificate directory for mutual TLS")
    ap.add_argument("--port-file", default=None)
    ap.add_argument("--out", default=os.path.join("demo", "_out", "server_summary.json"))
    ap.add_argument("--linger", type=float, default=5.0,
                    help="stay up after the last round so clients can observe the final state")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.shards, "meta.json"), encoding="utf-8"))
    learner = MaskAwareMLP(hidden=tuple(meta["hidden"]))
    theta0 = learner.initial_state(meta["n_features"], seed=0)

    coord = ServerCoordinator(experiment_id="demo", clients=meta["clients"],
                              schema=meta["schema_id"], round_timeout_s=args.timeout,
                              on_timeout=args.on_timeout)
    app = ServerApp(coord)

    ctx = None
    if args.tls:
        from mechfedgnn.security import server_context
        ctx = server_context(args.tls)
    httpd = serve(app, args.host, args.port, ssl_context=ctx)
    port = httpd.server_address[1]
    scheme = "https" if ctx else "http"
    print(f"server listening on {scheme}://{args.host}:{port} "
          f"({'mutual TLS' if ctx else 'DEV header identity - not a security control'})",
          flush=True)
    if args.port_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.port_file)), exist_ok=True)
        with open(args.port_file, "w", encoding="utf-8") as f:
            f.write(str(port))

    # every client starts from the same theta0; after round 1 each has its own
    assignments = {c: theta0 for c in meta["clients"]}
    summary = {"rounds": [], "rejections": app.rejections}
    try:
        for r in range(1, args.rounds + 1):
            coord.open_round(r, assignments)
            print(f"round {r}: open, waiting for {len(meta['clients'])} clients", flush=True)
            while coord.missing():
                coord.check_deadline()          # raises explicitly on timeout
                time.sleep(0.05)

            def weights_for(client, counts, _order=meta["clients"]):
                """Sample-size (FedAvg) weighting over DECLARED counts, through the
                same policy objects the local research code uses."""
                sizes = np.array([counts[c] for c in _order], float)
                i = _order.index(client)
                w, gamma, _ = policy_for("fedavg", {"gamma": 0.5, "alpha": 1.0,
                                                    "beta": 1.0}).weights({}, sizes / sizes.sum(), i)
                return w, gamma

            assignments, record = coord.close_round(weights_for)
            print(f"round {r}: closed; " + ", ".join(
                f"{x['receiver']}->{x['new_version'][:8]}" for x in record), flush=True)
            summary["rounds"].append(record)
    except RoundTimeout as e:
        summary["timeout"] = str(e)
        print(f"ROUND TIMEOUT: {e}", flush=True)
    finally:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=str)
        print(f"wrote {args.out}", flush=True)
        # Clients poll for the final round's completion; exiting immediately
        # would refuse their last request. Linger, then stop.
        time.sleep(max(0.0, args.linger))
        httpd.shutdown()
        print("server stopped", flush=True)


if __name__ == "__main__":
    main()
