# Validation report

**159 tests pass** (86 pre-existing research tests + 73 added by the refactor).
Run: `python -m pytest tests -q` — or `-m "not slow"` to skip the one test that
spawns real processes.

Passing these tests does **not** mean production readiness. See
`docs/SECURITY.md` §3 for what is explicitly not protected.

## Research correctness (preserved behaviour)

| Requirement | Test | Result |
|---|---|---|
| Signature fixtures | `test_signatures.py` (pre-existing), `test_golden_equivalence.py::test_signatures_are_unchanged` | exact equality of `r`, `H`, `J`, `C`, histograms for all three builders |
| Weight normalization and fallbacks | `test_golden_equivalence.py::test_weights_gamma_and_fallback_are_unchanged` | identical weights, gamma and fallback reason under **equal and non-uniform** sizes |
| FedAvg reduction | `test_kernel.py::test_fedavg_limit_unequal_sizes`, golden aggregation | α=0, β=1, γ=p_i reproduces `Σ p_j θ_j` to 1e-8 |
| Parameter-schema compatibility | `test_golden_equivalence.py::test_aggregator_rejects_incompatible_and_invalid_weights` | mismatched shapes, non-zero receiver weight and non-unit sums all rejected |
| Duplicate-group split separation | `test_corrections.py`, `test_design.py` | no duplicate group crosses design/client/fold boundaries |
| Existing mask invariants | `test_injector.py`, `test_matched.py`, `test_corrections.py` | null, dose-response, rate calibration, M0↔M1 exact counts |
| End-to-end equivalence with the original code | `test_golden_equivalence.py::test_end_to_end_metrics_and_weights_are_unchanged` | 432 (E1) and 384 (E5) metric rows plus realised weights match the captured reference |
| Scores and their undefined decisions | `test_golden_equivalence.py::test_scores_and_none_decisions_are_unchanged` | every registered score, and every `None`, unchanged |

Golden fixtures were captured from the **original** implementation
(`tools/capture_reference.py`) before extraction, on **synthetic** data so a
fresh clone can run them without downloading anything.

## Network execution

| Requirement | Test | Result |
|---|---|---|
| Independent clients complete a round | `test_federation.py::test_independent_client_processes_complete_rounds` (slow) | 1 server + 3 client **processes**, 2 rounds, all exit 0 |
| In-process and network produce matching aggregation | `test_in_process_and_network_transports_produce_identical_aggregation` | identical to 1e-12 under **non-uniform** weights |
| Stale update rejected | `test_stale_update_for_a_closed_round_is_rejected` | `stale_or_late_round` |
| Duplicate update not applied twice | `test_duplicate_update_cannot_be_applied_twice` | `duplicate_update`; one stored update |
| Client timeout follows the declared policy | `test_timeout_is_explicit_and_does_not_change_participants` | both `fail` and `pause` raise explicitly; participants untouched |
| Personalised responses reach the correct receiver | `test_personalised_parent_is_validated_per_client`, `test_close_round_produces_one_personalised_model_per_client` | c1's parent version is rejected for c0; one distinct model per receiver |

## Security

| Requirement | Test | Result |
|---|---|---|
| Unknown client certificate rejected | `test_unknown_client_certificate_is_rejected` | handshake fails (different CA) |
| Incorrect server certificate fails the connection | `test_incorrect_server_certificate_causes_connection_failure` | client verification fails |
| Certificate/request identity mismatch rejected | `test_certificate_and_request_identity_mismatch_is_rejected` | `identity_mismatch` (409) |
| Not enrolled in this experiment | `test_authenticated_identity_not_enrolled_in_this_experiment_is_refused` | 403 |
| Oversized, malformed and non-finite payloads rejected safely | `test_oversized_malformed_and_non_finite_payloads_are_rejected_before_aggregation` | 400; **nothing reached aggregation** |
| Uploaded state cannot execute code | `test_payload_cannot_carry_executable_objects` | pickle payload refused (`allow_pickle=False`) |
| Logs contain no credentials or raw records | `test_logs_and_error_bodies_carry_no_payload_values_or_key_material` | no payload values, no key material |
| Keys never tracked | `test_no_credential_like_files_are_tracked_by_git` | zero tracked `.key/.pem/.crt/.p12/.pfx` |

## Recovery

| Requirement | Test | Result |
|---|---|---|
| Interrupt at a controlled boundary and resume | `test_interrupted_open_round_resumes_without_applying_an_update_twice` | 2 of 3 reported; resumed coordinator still reports `c2` missing |
| No update applied twice | same test | the identical retry is rejected as `duplicate_update` |
| Complete vs incomplete round distinguished | `test_completed_round_resumes_at_the_next_round`, `test_resumed_completed_round_does_not_reapply_its_updates` | open → resume round 1; complete → start round 2 with aggregated models unchanged |
| Atomic writes | `test_checkpoint_write_is_atomic_and_leaves_no_partial_files` | no `.tmp` remains; header is the commit point |
| Corruption detected | `test_tampered_or_missing_model_file_is_detected` | version mismatch and missing file both raise |

## Reproducibility (Milestone B)

`test_reproducibility.py` — 15 tests: schema defaults and per-key rejection;
legacy seed derivation **bit-identical** to `loop._seed` while independent
streams move only the stream that changed; dataset checksum changes when one
value changes; the store refuses to overwrite a completed run; model states load
without `allow_pickle`; a genuinely NaN-holed dataset trains end to end with
injection **disabled**; provenance records the dirty flag, checksums, split and
mask ids, preprocessing and where it was fitted, realised weights, fallbacks and
participation.

## Bugs found and fixed during the refactor

1. **Golden fixture rounding** — capture rounded to 12 decimals while the test
   compared exactly (4.8e-13 drift). Fixed by storing full precision; the
   comparison stays exact.
2. **Demo client raced the server** — the server opens round *r+1* immediately
   after closing *r*, so polling for "round *r* closed" could never match.
   Fixed by treating "the round has advanced past *r*" as completion.
3. **Server exited before clients could confirm** the final round. Fixed with a
   linger period.
4. **Open-round checkpoints saved update ids but not payloads** — every resend
   would be rejected as a duplicate while the payload was gone, so a resumed
   round could never complete. Fixed by checkpointing the received updates too.

## Known limitations

- The multi-round demonstration verifies the framework; it is **not** a
  scientific result, and the historical single-round protocol is unchanged.
- No secure aggregation, differential privacy or Byzantine robustness.
- `cryptography` is a new dependency, used only for development certificates.
- The one process-spawning test is marked `slow`; it adds ~25 s.
