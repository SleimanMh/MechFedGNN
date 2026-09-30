# MechFedGNN — architecture audit and refactoring plan

*Engineering phase. No methodology change, no tuning, no new research sweeps.
Existing findings, raw data and resolved experiment configs stay unchanged.*

**Standing research result this framework must preserve:** across E1, E1M, the
§16 correction study and E5, the fixed scores showed **no meaningful advantage
over equal donor weighting** (240 corrected + 320 E5 contrasts, all inside the
declared ±2 % margin). The framework must keep those runs reproducible and make
*future* comparisons cheaper — not make the old ones look different.

## 1. What exists today

~2,600 lines of research Python, 86 passing tests, six CLI entry points, and a
committed results record (`docs/RESULTS.md`, `docs/PROTOCOL_E5.md`,
`docs/E5_DIAGNOSTIC.md`).

| Module | Contents | Verdict |
|---|---|---|
| `data/design.py` | `load_dataset`, `design_split`, `duplicate_groups`, `representatives`, `decile_edges` | reuse, split responsibilities |
| `data/inject.py` | `build_roles`, `form_panels`, `ordering_probs`, `inject`, `expected_feature_rates` | reuse as injector |
| `data/clients.py` | `build_clients`, population assignment, `_split`, `relocate_duplicates`, group profiles, `size_check` | **entangled — extract** |
| `data/matched.py` | E1M construction (pi_A/pi_B, exact counts) | reuse as an injector/partitioner variant |
| `data/e5.py` | E5 construction (P0/P1 x M0/M1) | reuse as an injector/partitioner variant |
| `signatures.py` | `rates`, `joint_absence`, `joint_observation`, `phi`, `chi2`, `signature`, `histograms`, `population_similarity` | **reuse as-is** (signature provider) |
| `scores.py` | `w_h`, `w_c`, `w_marginal`, `missingness_similarity`, `rate_similarity`, `combined_q` | reuse; needs a registry |
| `kernel.py` | `donor_weights`, `aggregate` | **reuse as-is**; already two functions, needs two interfaces |
| `model.py` | `MaskMLP`, `Standardiser`, `train`, `train_early_stopping`, `predict`, `metrics` | reuse; preprocessing must move out |
| `loop.py` | `run_seed` (119 lines) + scoring + weighting + scaler policy | **the main extraction target** |
| `headroom.py`, `geometry.py`, `stats.py` | diagnostics and paired-interval statistics | reuse as-is (evaluator) |
| `report.py`, `report_tables.py` | provenance, CSVs, REPORT.md | split: artifact store vs presentation |
| `run_e1/e1m/e5.py`, `pilots.py`, `check_e1m/e5.py` | CLI entry points | keep as thin wrappers |

## 2. Where responsibilities are mixed

1. **`loop.run_seed` is the orchestrator and almost everything else.** In one
   function: client construction dispatch, local training, signature
   computation, pairwise scoring, arm weighting, aggregation, adaptation,
   evaluation at multiple folds/budgets, geometry logging, functional logging,
   headroom, compute accounting, and output-row shaping. Every new experiment
   so far was added by threading another flag through it (`builder=`,
   `cfg["e5"]["s_exclude"]`, `cfg["arms"]`).
2. **Scoring and weighting are conflated in `arm_weights`.** One function picks
   *which score* (`SCORE_OF_ARM` lookup), decides the *fallback*, and computes
   the *normalised weights*. The brief requires these separate.
3. **`pair_scores` computes every score eagerly** from a hard-coded list,
   regardless of which arms are active — no registry, so adding a score means
   editing three places (`pair_scores`, `SCORE_OF_ARM`, and the score-name
   literal inside `run_seed`).
4. **Preprocessing policy is split across modules.** `Standardiser` lives in
   `model.py`; the policy that decides *shared vs per-client* fitting lives in
   `loop.shared_scaler`/`scaler_for`. Where it was fitted is not recorded.
5. **`build_clients` fuses four policies** — population assignment, duplicate
   relocation, fold splitting, and mask injection — so a client with naturally
   incomplete data cannot skip injection.
6. **`report.write_run` is provenance + storage + presentation.** `run_id`
   records date, git sha and a config hash, but **not** dirty-tree state,
   dataset checksums, or split/mask identifiers.
7. **One seed stream for everything.** `_seed(seed, purpose, k)` derives
   partitioning, masks and training from a single root; Milestone B requires
   three separate streams.
8. **No client/server/transport layer exists at all** — entirely new code.

## 3. Target architecture

Layered; each arrow is a call, and no lower layer imports an upper one.

```
              +---------------- experiments/ (CLI wrappers) ----------------+
              |  run_e1  run_e1m  run_e5  pilots  check_*  (unchanged args) |
              +--------------+--------------------------+-------------------+
                             |                          |
                 +-----------v----------+   +-----------v-----------+
                 |  ServerCoordinator   |<->|      Transport        |
                 |  rounds, aggregation |   | in-process | mTLS gRPC|
                 +-----------+----------+   +-----------+-----------+
                             |                          |
                             |              +-----------v----------+
                             |              |    ClientRuntime     |  (one per client)
                             |              +-----------+----------+
     +-----------------------+------------------+-------+------------+
     |                       |                  |                    |
+----v-----------+ +---------v---------+ +------v-------+ +----------v------+
| ScoringStrategy| | WeightingPolicy   | | Aggregator   | |  LocalLearner   |
| summaries->    | | blend / sharpen / | | states + w   | | train / adapt / |
| score          | | normalise /       | | -> new state | | predict /export |
|                | | fallback          | |              | |                 |
+----+-----------+ +-------------------+ +--------------+ +--------+--------+
     |                                                             |
+----v------------+ +-------------+ +--------------+ +----------+ +v---------+
|SignatureProvider| |DataProvider | |Split/Preproc | | Injector | |Evaluator |
| r,H,C,J,hists   | | per-client  | | dup groups,  | | optional | | metrics +|
|                 | | records only| | roles,scaler | | + checks | | transfer |
+-----------------+ +------+------+ +--------------+ +----------+ +----------+
                           |
                   +-------v--------+
                   | ArtifactStore  |  config, provenance, metrics,
                   |                |  weights, checkpoints
                   +----------------+
```

**The three operations stay separate**, as required:

| Operation | Interface | Current home |
|---|---|---|
| Score calculation | `ScoringStrategy.score(receiver_summary, donor_summary) -> float or None` | `scores.py` + `loop.pair_scores` |
| Weight normalization | `WeightingPolicy.weights(scores, sizes, receiver_idx) -> (w, gamma, fallback)` | `kernel.donor_weights` + `loop.arm_weights` |
| Parameter aggregation | `Aggregator.combine(states, receiver_idx, w, gamma) -> state` | `kernel.aggregate` |

### Registered methods (definitions preserved exactly)

| Registry key | Score | Source |
|---|---|---|
| `local-only` | — | no aggregation |
| `fedavg` | — | alpha=0, beta=1, gamma=p_i |
| `uniform-donor` | — | equal donor weights, gamma shared with score arms |
| `marginal-rate` | `0.5(1+cos(r_i, r_j))` — **symmetric similarity** | `scores.rate_similarity` |
| `coverage-marginal` | `sum r_i(1-r_j) / sum r_i` — **directed coverage** | `scores.w_marginal` |
| `missingness-similarity` | `0.5(1+cos(vec C_i, vec C_j))` | `scores.missingness_similarity` |
| `coverage-W_H` | `sum H_i*J_j / sum H_i` | `scores.w_h` |
| `coverage-W_C` | `sum [C_i]+ * J_j / sum [C_i]+` | `scores.w_c` |
| `population-S` | histogram overlap | `signatures.population_similarity` |
| `combined-Q` | `(1-lambda) W_H + lambda S` | `scores.combined_q` |
| `combined-Q-marginal` | `(1-lambda) W_marg + lambda S` | `scores.combined_q` |

`marginal-rate` (symmetric) and `coverage-marginal` (directed) are **different
methods** and stay distinct keys.

### Conventions carried into interface docstrings

`M = 1` observed; `H` joint absence; `C` binary association (phi); `J` joint
observation; constant-feature rows/cols of `C` set to 0 with rates kept
separately; undefined score -> `None` -> declared sample-size fallback
(CLAUDE.md 7.1); self-weight `gamma`, sample-size weights `p`, and donor
weights `w` remain three separate quantities; the FedAvg limit
(`alpha=0, beta=1, gamma_i=p_i`) is unit-tested to 1e-8. **No claim is made
that any MCAR mask automatically recovers FedAvg.**

The active learner stays the mask-aware MLP with prediction loss only.
`LocalLearner` gets an extension point; **no GNN, reconstruction head or
attention is implemented.**

## 4. Historical-command compatibility

All six entry points keep their current names and flags and become thin
wrappers over the new orchestrator. No historical command changes:

| Command | Status |
|---|---|
| `python run_e1.py [--datasets ...] [--corrected] [--assignment corr|random]` | unchanged |
| `python run_e1m.py [--datasets ...] [--conditions a b c] [--corrected]` | unchanged |
| `python run_e5.py --stage dev|eval [--datasets ...] [--conditions ...] [--s-config ...]` | unchanged |
| `python pilots.py [--headroom-only|--power]` | unchanged |
| `python check_e1m.py`, `python check_e5.py` | unchanged |
| `python -m analysis.*` | unchanged |

Any command that must change will get an explicit old->new mapping in
`docs/MIGRATION.md`. Existing run directories under `results/` are never
rewritten; new runs go to new directories.

**Per-experiment checks stay per-experiment.** `check_e1m` keeps its
cross-client matched-rate tolerance (E1M matches marginals by design);
`check_e5` keeps its M0<->M1 per-client count invariant (E5 deliberately gives
clients *different* rates). The framework must not apply one globally — that
error was already made once and corrected (`docs/E5_DIAGNOSTIC.md` section 4).

## 5. Validation required before replacing each implementation

Milestone A captures golden reference cases from the current code **before**
extraction, then asserts equality after. Ordered by what breaks most quietly:

| Replaced component | Required evidence |
|---|---|
| Data provider / split policy | identical client row indices and fold indices per seed; duplicate groups never split |
| Injector | identical masks, bitwise, for E1 / E1M / E5 builders |
| Signature provider | identical `r`, `H`, `J`, `C`, histograms (exact) |
| Scoring strategy | identical score values **and** `None` decisions |
| Weighting policy | identical weights, gamma, and fallback reason — including a **non-uniform** fixture |
| Aggregator | identical parameters (atol 1e-8); FedAvg limit test still passes |
| Local learner | identical predictions/metrics within tolerance, same seeds/batches/init/optimizer |
| Orchestrator | small end-to-end run matches saved metrics rows |

A non-uniform-weight fixture is mandatory: with four equal-sized clients,
`uniform-donor` and `fedavg` produce near-identical weights, so a transport or
aggregation bug would pass a uniform-only test.

Numerical policy: exact equality for integers, indices, masks and signatures;
`atol=1e-8` for parameters and aggregation; declared tolerance for
trained-model metrics. **Bitwise equality across different hardware is not
required.**

## 6. Milestones

| Milestone | Scope | Exit criterion |
|---|---|---|
| **A** | Golden references; extract interfaces behind current behaviour | refactored local execution reproduces references without rerunning the research suite |
| **B** | Validated config, full provenance, separate RNG streams, optional injection | a run records revision + dirty flag, checksums, split/mask ids, scores **and realised weights**, fallbacks, participation |
| **C** | 1 server + 3 client processes, short multi-round demo, in-process and network parity | stale/duplicate/timeout policies tested; personalised responses reach the right receiver |
| **D** | mTLS, identity bound to certificate, safe deserialisation, payload validation | unknown cert rejected; identity mismatch rejected; malformed/oversized/non-finite rejected |
| **E** | Atomic checkpoints, documented recovery | resume applies no update twice; complete vs incomplete round distinguished |

Milestone C's multi-round demo **verifies the framework; it is not a new
scientific result.** The historical single-round protocol stays available and
unchanged.

## 7. Out of scope for this phase

Secure aggregation, differential privacy, Byzantine robustness, GNN/attention/
reconstruction learners, new sweeps, methodology changes, and the donor-utility
estimability study recommended in `docs/E5_DIAGNOSTIC.md` section 6.
