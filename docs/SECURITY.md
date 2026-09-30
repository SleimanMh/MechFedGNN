# Threat model and transport security

Scope: the separate-process federation demonstration. This is a **research
framework**, not a deployable clinical system. Everything below describes what
is and is not protected.

## 1. Threat model

| Threat | Mitigation here | Residual risk |
|---|---|---|
| **Unauthorized participation** — a stranger joins a round | Mutual TLS: the client must present a certificate chaining to the experiment CA. Identity = certificate Common Name. An `Allowlist` then binds that identity to the experiments it may join. | Anyone holding a valid client key *is* that client. Key distribution and revocation are out of scope (no CRL/OCSP). |
| **Interception** — reading updates in flight | TLS ≥ 1.2 with certificate verification on both ends; the client checks hostname and chain. | An attacker with the server's private key, or a trusted-CA compromise, can read traffic. |
| **Replay** — resending a captured update | Each update carries an `update_id` (client, round, content). A repeated id in the same round is rejected as `duplicate_update`; an update for a closed or different round is `stale_or_late_round`; the `parent_version` must match **that client's own assignment**. | Within a single open round, a captured update from the same client that was never delivered could be relayed by an attacker who also holds the client's certificate. |
| **Malformed payloads** — crashing or exploiting the server | Size cap before parsing; strict envelope typing; `np.load(allow_pickle=False)` so no object can be deserialized; tensors must be floating point, finite, and match the declared schema. Rejection happens **before aggregation**. | A structurally valid but statistically poisonous update is still accepted (see §3). |
| **Accidental disclosure** — data leaking through logs or errors | Request bodies are never logged; error responses carry a reason code, not payload contents; model states are stored as `.npz` arrays; keys and certs are gitignored and tested for. | Aggregate mask statistics themselves carry information (see §3). |

## 2. What is implemented

- **TLS with certificate verification**, minimum TLS 1.2 (`ssl`, OpenSSL).
- **Mutual TLS** for the controlled deployment: `verify_mode = CERT_REQUIRED`
  on the server, so a client certificate is mandatory.
- **Identity bound to the certificate.** The authenticated identity is the peer
  certificate's Common Name. A `client_id` in a message body is **never trusted
  on its own** — the server compares the two and rejects a mismatch
  (`identity_mismatch`).
- **Authorisation separate from authentication.** `Allowlist` maps identity →
  permitted experiments; being authenticated is not being enrolled.
- **Locally generated development certificates** (`make_dev_certs`): a local CA
  plus per-client leaves, short-lived, Common Name and Organization marked
  `MechFedGNN-DEV-DO-NOT-USE-IN-PRODUCTION`.
- **Secrets excluded from version control**: `demo/_certs/`, `*.pem`, `*.key`,
  `*.crt`, `*.p12`, `*.pfx`, `secrets.*`, `.env` are gitignored, and a test
  asserts no credential-like file is tracked.
- **Payload size limits and timeouts** on both transport and application sides.
- **Round and version validation**, including personalised parent models.
- **Safe model-state serialization** — arrays only, never pickle.
- **Tensor name, shape, dtype and finite-value checks** before aggregation.

No custom cryptography is implemented. Only `ssl` (OpenSSL) and `cryptography`
are used.

## 3. Documented limits — read before drawing conclusions

- **The server can inspect every transmitted model update and summary.** There
  is no secure aggregation and no trusted execution. The server is trusted with
  what it receives.
- **Encryption is not privacy.** TLS protects data *in transit*. It provides
  **no secure aggregation and no differential privacy**; neither is implemented.
- **An authenticated client can still send a malicious but structurally valid
  update.** Validation rejects malformed, oversized, non-finite and
  schema-mismatched payloads — it does not detect poisoning. No
  Byzantine-robust aggregation is implemented.
- **Aggregate mask statistics can reveal information.** `r`, `H`, `J`, `C` and
  population histograms are computed over a client's training rows; with small
  folds or extreme rates they can narrow down individual records. They are
  aggregate, not anonymous.
- **Process separation is a software boundary, not isolation.** Running clients
  as separate processes shows the code respects the boundary. Anyone with
  administrative access to the host can read every shard. If containers are
  used, give each client its own mount.
- **Development certificates are unsuitable for production**: self-signed local
  CA, unencrypted private keys on disk, no revocation, short lifetime.
- **Clinical deployment and regulatory compliance are outside this
  demonstration.** Passing these tests does not make the system production-ready.

## 4. Generating development certificates

```python
from mechfedgnn.security import make_dev_certs
make_dev_certs("demo/_certs", ["c0", "c1", "c2"])   # CA + server + per-client leaves
```

This writes `ca.pem`, `ca.key`, `server.pem/.key`, `client_<id>.pem/.key` and a
`README.txt` marking them development-only. The directory is gitignored.

## 5. Secure launch

```bash
python -m demo.prepare_shards
python -c "from mechfedgnn.security import make_dev_certs; make_dev_certs('demo/_certs', ['c0','c1','c2'])"

python -m demo.server --rounds 3 --port 8443 --tls demo/_certs \
    --port-file demo/_out/port.txt

# each client, in its own process (and ideally its own container/mount):
python -m demo.client --client-id c0 --url https://127.0.0.1:8443 --rounds 3 --tls demo/_certs
python -m demo.client --client-id c1 --url https://127.0.0.1:8443 --rounds 3 --tls demo/_certs
python -m demo.client --client-id c2 --url https://127.0.0.1:8443 --rounds 3 --tls demo/_certs
```

With `--tls`, identity comes from the certificate. **Without `--tls`** the
server falls back to a development `X-Dev-Client-Id` header, which is
explicitly **not a security control** and exists only so the in-process and
plain-HTTP paths can be exercised in tests.
