# Separate-process federation demonstration

One server and three independent client processes exchange model updates over
the shared message protocol. **This verifies the framework's software
boundaries; it is not a new scientific result.** The historical single-round
research protocol (`run_e1.py`, `run_e1m.py`, `run_e5.py`) is unchanged.

## Run it

```bash
python -m demo.prepare_shards                 # one data shard per client
python -m demo.server --rounds 3 --port 8443 --port-file demo/_out/port.txt
# in three more terminals:
python -m demo.client --client-id c0 --url http://127.0.0.1:8443 --rounds 3
python -m demo.client --client-id c1 --url http://127.0.0.1:8443 --rounds 3
python -m demo.client --client-id c2 --url http://127.0.0.1:8443 --rounds 3
```

With mutual TLS (see `docs/SECURITY.md`), add `--tls demo/_certs` to both the
server and every client.

## What crosses the boundary

| Direction | Carries |
|---|---|
| server -> client | that client's **own** assigned parent model, its version, the schema id |
| client -> server | a model update, the **declared sample count**, and (when the method needs it) an **aggregate** mask signature |

Never transmitted: features, labels, per-row masks, per-example predictions.

Each client process opens **only its own shard** (`ClientRuntime.load` refuses a
shard belonging to another client). The server process holds no data provider.

## Policies

| Situation | Behaviour |
|---|---|
| A client has not reported | the round stays open; `missing()` names it |
| Deadline passes | `RoundTimeout`, **explicitly** fail or pause; participants are never silently changed and weights are never renormalised under a different rule |
| Same update sent twice | rejected as `duplicate_update`; applied at most once |
| Update for a closed or wrong round | rejected as `stale_or_late_round` |
| Wrong parent model version | rejected as `parent_version_mismatch` — validated against **that client's own assignment**, since clients hold different models |
| Body `client_id` differs from the authenticated identity | rejected as `identity_mismatch` |

The first demonstration requires **all** configured clients to complete a round.

## Process separation is a software boundary

Running clients as separate processes demonstrates that the code respects the
boundary. It is **not** isolation against the host administrator: anyone with
access to this machine can read every shard. If containers are used, give each
client its own mount.
