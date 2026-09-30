# MechFedGNN — project overview

*One document covering how the work developed, what each quantity we compute
actually affects, how the code is now organised, and how to run and present it.*

- **§1** — how to run everything
- **§2** — the development process, experiment by experiment, with the reason for each decision
- **§3** — every metric we compute, and what it does and does not affect
- **§4** — the refactored architecture and how the components connect
- **§5** — the dashboard, screen by screen

Detailed evidence lives in `docs/RESULTS.md` (3,000 lines of run output),
`docs/PROTOCOL_E5.md`, `docs/E5_DIAGNOSTIC.md`, `docs/ARCHITECTURE.md`,
`docs/VALIDATION.md` and `docs/DASHBOARD.md`. This file is the guided path
through them.

---

# 1. Running everything

## 1.1 Setup, once

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

On macOS or Linux the activate line is `source .venv/bin/activate`.

## 1.2 The dashboard — the thing to show a person

```bash
python -m dashboard.app --allow-launch
```

Open <http://127.0.0.1:8765>. Without `--allow-launch` everything still works
except starting new runs. It binds localhost deliberately: there is no
authentication.

Replay needs completed runs under `results/`. If that directory is empty, the
Live tab still works on its own — it generates its own synthetic data.

## 1.3 The test suite

```bash
python -m pytest tests -q
```

`-m "not slow"` skips the three tests that spawn real client processes (~70 s
saved). The suite is the honest summary of what is verified; `docs/VALIDATION.md`
maps each requirement to the test that holds it.

## 1.4 A real federated run, as separate processes

Three terminals, or one with the dashboard's Live tab doing it for you.

```bash
python -m demo.prepare_shards --out demo/_shards --clients 3
```

```bash
python -m demo.server --shards demo/_shards --rounds 2 --port 8443 --tls demo/_certs
```

```bash
python -m demo.client --client-id c0 --url https://127.0.0.1:8443 --tls demo/_certs --send-signature --evaluate-fold val
```

Repeat the last line for `c1` and `c2`. Certificates are minted by
`mechfedgnn.security.make_dev_certs`; the Live tab does this for you.

## 1.5 Reproducing the research experiments

These need the UCI tables, which are gitignored.

```bash
python -m data.download_uci
```

```bash
python check_e5.py
```

```bash
python run_e1.py --corrected --assignment corr
```

```bash
python run_e1m.py --corrected --assignment corr
```

```bash
python run_e5.py --stage eval
```

```bash
python -m analysis.build_results_md
```

The last command rebuilds `docs/RESULTS.md` from the saved artifacts. Every
historical command kept its name and flags through the refactor — see
`docs/MIGRATION.md`.

---

# 2. How the work developed

## 2.0 The question

Clients hold tabular data and are **incomplete in different ways** — not just
different amounts of missing data, but *different columns* missing, with
different co-occurrence structure. Each client can summarise its own missingness
and its own population cheaply and without disclosing records.

**The hypothesis:** those summaries predict which other clients are useful
collaborators, so a server can turn them into personalised aggregation weights
and beat weighting everyone equally.

Everything below is the attempt to test that honestly.

## 2.1 The methodological decisions made before any result existed

These matter more than any individual number, because they are what makes the
negative result trustworthy.

| Decision | Why |
|---|---|
| **Effect size declared in advance: δ = 2 % relative RMSE** | Without a pre-declared margin, "no difference" and "a difference we could not detect" are indistinguishable, and any result can be narrated after the fact. |
| **The seed is the unit of replication** | Receivers within a seed share data, donors and the same starting model. Treating four receivers as four independent observations would quadruple the apparent sample and shrink every interval. |
| **A three-way decision: meaningful / negligible / unresolved** | A wide interval that happens to exclude zero is not evidence. "Unresolved" means *we did not demonstrate a difference*, not *there is none*. |
| **`uniform-donor` is the reference arm** | The interesting question is not "does collaboration help" but "does the *score* help". Equal donor weighting is the thing a score has to beat to be worth its complexity. |
| **A declared fallback, written before running** | If a score is undefined for any donor, or all donors tie within 1e-9, it cannot rank donors, so the arm falls back to sample-size weighting and records that it did. Without this rule, an undefined score becomes an ad-hoc decision made under pressure. |
| **Datasets are never pooled** | Four datasets are four experiments. Pooling them would let one dataset's behaviour carry the conclusion. |

## 2.2 Stage 0–3: construction, and what it cost to get it right

**Stage 0 — the injector.** Missingness is injected in *panels* (groups of
features that go missing together), so the data has real joint structure to
detect rather than independent per-cell holes. Validated before use with a null
check (no panel structure produces φ ≈ 0), a dose-response check (more
missingness produces proportionally more), and rate calibration (the realised
rate matches the requested one). Realised rates: 0.294–0.307 against a requested
0.300.

**Stage 1 — signatures, scores, aggregation kernel.** Written tests-first. The
key test is the **FedAvg reduction**: with α = 0, β = 1, γ_i = p_i the whole
personalised machinery must reproduce plain FedAvg to 1e-8. If it cannot, the
method is not a generalisation of the baseline and comparisons mean nothing.

**Stage 2 — clients, model, orchestration.** This stage *failed first* and was
committed as it stood before being fixed, because the record of a setup that did
not work is part of the evidence.

Four problems and their fixes:

| Problem | Fix | Why it mattered |
|---|---|---|
| `receiver_fraction` made receivers systematically different from donors | removed | it confounded "receiver vs donor" with "different missingness" |
| Design split was a fraction, so small datasets lost too much | flat 10 %, 150-row floor | protein and kin8nm were donating far more rows than concrete |
| Headroom checked with a fixed budget | early stopping on the same validation rule | a "pooling helps" verdict that is really "pooled got more steps" is worthless |
| Local learning curve normalised across datasets per step | normalise each dataset by its own minimum across steps | **this was a real bug**: it tracked the wine/concrete scale ratio and chose 10 local steps. Fixed, the answer is 200 |

**Dataset preconditions, applied before seeing any result.** Rejected: *naval*
(maskable features had median |corr| 0.995 — masking them removes nothing),
*energy* (92 training rows per client at K = 4), *power* (only one panel could be
formed). Kept: concrete, wine, kin8nm, protein. Rejecting datasets *after*
seeing results would be selection; the preconditions were fixed first.

## 2.3 E1 — the main signal comparison

Four datasets, K = 4 clients in two latent missingness groups, 10 seeds, 9 arms.
Panel-based MAR missingness, `driver_overlap` 0.5.

**Result 1 — the scores made no practical difference.** All 48 score-arm vs
`uniform-donor` contrasts fall entirely inside ±2 %. The widest bound across all
of them is 1.44 %; every point estimate is within ±0.40 %.

**Result 2 — collaborating at all matters, in both directions.**

| Dataset | Every mixing arm vs `local-only` | Receivers harmed by every mixing arm |
|---|---|---|
| wine | **−6.6 to −7.5 %**, meaningful | 0 / 4 |
| concrete | −3.8 to −4.1 % | 0 / 4 |
| kin8nm | +1.0 to +1.9 % (harm) | 3 / 4 |
| protein | +0.4 to +0.8 % (harm) | 4 / 4 |

So collaboration is worth several percent on some datasets and actively harmful
on others — and **no weighting rule avoided the harm**. That asymmetry is the
real finding of E1, and it is about *whether* to collaborate, not *with whom*.

**Result 3 — an oracle can pick a good donor; validation cannot.** The
test-picked best single donor beats the uniform mixture by up to 3–4 % on
concrete. The validation-picked donor never does. The opportunity exists; it is
not identifiable from the data a client actually has.

**Result 4 — why weighting barely moved anything (geometry).** We checked
whether the scores fail because all clients are basically identical. They are
not: pairwise parameter distances are 0.67–1.23× each client's distance from the
starting model. But every score-weighted mixture lands 0.00–0.34 from the uniform
mixture, while single-donor mixes sit 0.50–1.72 from the centroid. **Averaging
with weights in the range these scores produce barely moves the model.** The
scores are not wrong so much as too gentle to matter.

**Result 5 — a prediction we made, and it failed.** We predicted that
missingness-similarity `s` would recover the injected client groups better than
marginal rates alone. Marginal-rate similarity alone recovers them perfectly
(ARI = 1.0 on all four datasets), so the prediction could not be supported.
Recorded as failed rather than dropped.

## 2.4 E1M — removing the easy explanation

E1 left an objection open: maybe the scores looked useless because marginal
rates already carried all the signal, so nothing was left for joint structure.

E1M removes that by construction: marginal rates are **flat across clients by
design** — only the *joint* missingness structure distinguishes them. Three
conditions: (a) cell-wise masking (a genuine null), (b) independent panels,
(c) coupled panels.

**Result: all 72 contrasts inside ±2 %.** The scores do not help even when the
joint structure is the only thing distinguishing clients. The
"marginals-explain-everything" objection does not rescue the hypothesis.

E1M also reproduced the oracle-vs-validation gap **in the null condition (a)**,
where by construction there is no structure to find. That is what shows the gap
is about the *estimability of donor usefulness*, not about the scores.

## 2.5 The §16 correction study — a review found three construction flaws

A review found three problems affecting all results so far:

1. **Exact-duplicate records** — wine 28.8 % of rows, protein 6.9 %, concrete
   3.5 %, kin8nm 0 %. Duplicates split across train and test leak.
2. **Per-client standardisation** — clients' parameters were being averaged
   while sitting in different numerical coordinates.
3. **Target-dependent maskable assignment** — which features could go missing
   was chosen using the target, so the masking could carry target information.

Everything was re-run with all three corrected, plus a target-independent
control. **Sections 0–7 of `docs/RESULTS.md` were left exactly as originally
run** — corrections are reported alongside, never edited over.

**What survived:**

- The central claim — score arms equivalent to equal donor weights — survives
  **everywhere**: all 8 corrected E1 runs, all 12 corrected E1M cells.
- Wine's collaboration gain survives **and grows** (−7.2 % vs −6.6 %). The
  duplicates had been inflating `local-only`, not the gain.

**What did not:**

- Concrete's decision labels shift (estimates move ≤ 0.6 points, but intervals
  now cross δ).
- kin8nm's "harm stays under δ" **does not hold** under the target-independent
  assignment.
- Corrected E1M on concrete and wine is **VOID** by the declared rate tolerance.

That last one deserves a note. The corrected runs broke a tolerance we had
declared in advance. Restricting that tolerance to training folds would have
rescued the failing cells — and that change would have been visible *only*
because it rescued them. **We accepted the void instead.** That is exactly what
pre-declaration is for.

## 2.6 E5 — population compatibility, the last live hypothesis

E1 and E1M held client populations homogeneous. So their weak result for
population similarity `S` did not establish that `S` fails under population
shift — it had never been tested under one.

E5 tests it directly: a factorial design crossing **population shift** (P0 =
homogeneous, P1 = shifted) with **missingness structure** (M0, M1), eight arms,
ten frozen evaluation seeds disjoint from E1/E1M, and a **marginal-only coverage
control** (`coverage-marginal`) that isolates what *pairwise* coverage adds over
marginal availability. The protocol was written and committed before any
evaluation run.

**Result: all 320 primary contrasts negligible.** Every interval inside ±2 %, at
both timepoints.

| Comparison | Question | Answer |
|---|---|---|
| `combined-Q` vs `coverage-W_H` | does population similarity add value? | no, everywhere |
| `combined-Q` vs `population-S` | does coverage add value? | no, everywhere |
| `combined-Q` vs `combined-Q-marginal` | does **pairwise** structure add beyond marginal + S? | no, everywhere |
| `population-S` vs `uniform-donor` | does the population signal help at all? | no, everywhere |

Population similarity did not improve prediction **even where the shift was real
and `S` could see it**. And pairwise coverage added nothing beyond marginal
availability plus `S` — which, if the pairwise machinery is the contribution, is
the sharpest negative result in the project.

One methodological note: an earlier check imported E1M's cross-client rate
tolerance into E5. That was wrong — E5 deliberately gives clients *different*
rates, so that check would forbid the heterogeneity the experiment exists to
create. It was caught and corrected **before any E5 prediction result existed**,
and is recorded as a mis-specification rather than a relaxation.

## 2.7 The E5 diagnostic — why, not just whether

A negative result is more useful with a diagnosis. This is a post-hoc analysis
of existing artifacts — no retraining, no tuning, no new runs — and it is
labelled **exploratory** throughout.

**Was there an opportunity to miss?**

| Dataset | `local-only` | random donor | **test-best donor (retrospective)** | validation-selected donor |
|---|---|---|---|---|
| concrete | +1.8 to +3.8 % | +0.3 to +0.9 % | **−2.1 to −3.2 %** | −1.2 to +0.5 % |
| wine | +3.7 to +5.9 % | +0.1 to +0.5 % | ≈ −1.5 % | −0.2 to +0.7 % |
| kin8nm | −0.4 % | ≈ 0 % | ≈ −0.6 % | −0.3 to −0.6 % |
| protein | −0.3 % | ≈ 0 % | ≈ −0.2 % | −0.2 to −0.3 % |

Three things follow:

1. **Uniform aggregation is barely better than picking a random donor**
   (+0.00 to +0.89 % everywhere). Whatever it achieves, it is not achieved by
   being a clever combination.
2. **A real single-donor opportunity exists only on concrete** (−2.1 to −3.2 %).
3. **Validation cannot find it.** Validation-best agrees with test-best only
   42–57 % of the time on concrete, against 33 % chance with three donors, and
   the two disagree on *whether to collaborate at all* 37–43 % of the time.

**Diagnosis: weak score-utility alignment plus an inability of validation to
identify good donors — not absence of donor opportunity.** Donor rankings agree
with measured transfer at or below chance. The scores are measuring something
real about missingness; that something is not predictive of transfer.

## 2.8 The standing result

Across E1, E1M, the correction study and E5:

> **240 corrected contrasts + 320 E5 contrasts, all inside the declared ±2 %
> margin.** The fixed mask-based scores show no meaningful advantage over equal
> donor weighting.

What *is* established positively:

- **Whether to collaborate matters a lot** — ±3 to 7 % depending on dataset, in
  both directions.
- **The decision is dataset-level, not donor-level**, in this setting.
- **Donor choice has real headroom on one dataset**, and validation cannot
  exploit it.

This is a negative result reported as one. It is more useful than a marginal
positive would have been, because it is decisive within its declared margin and
because the diagnostic says where the failure is.

---

# 3. What each metric affects

This is the section to walk a reviewer through. The key point: these quantities
are **not interchangeable**, they enter at **different stages**, and several of
them affect nothing at all in the current configuration — which is itself a
finding.

## 3.1 The pipeline, and where each quantity enters

```
  client's mask M
        |
        v
  [1] SIGNATURES      r, H, J, C, histograms      <- leaves the client (aggregate only)
        |
        v
  [2] SCORES          W_H, W_C, W_marg, s, rate, S, Q, Q_marg
        |                                          one number per (receiver, donor)
        v
  [3] WEIGHTS         base -> sharpen -> normalise -> w_j        (alpha, beta)
        |
        v
  [4] AGGREGATION     theta_new = gamma*theta_i + (1-gamma)*sum_j w_j*theta_j   (gamma)
        |
        v
  [5] ADAPTATION      a few local steps on the receiver's own rows
        |
        v
  [6] EVALUATION      RMSE / MAE / AUC        -> [7] Delta% and its interval
```

Scoring, weighting and aggregation are **three separate operations**. A new
score never requires touching weighting or aggregation. This separation is
enforced by the architecture (§4) and is why the dashboard can explain them as
distinct steps.

## 3.2 The signatures — what a client reveals

| Quantity | What it is | What it affects |
|---|---|---|
| **`r`** | per-feature missing rate | feeds `W_marg` and `rate`. The **marginal** view of missingness |
| **`H`** | joint-absence matrix: how often features f and g are missing *together* | feeds `W_H`. The receiver's "gaps" |
| **`J`** | joint-observation matrix: how often f and g are observed together | feeds `W_H` and `W_C`. The donor's "coverage" |
| **`C`** | binary association (φ) between missingness indicators | feeds `W_C` and `s`. Constant features get φ = 0 with the rate kept in `r` |
| **histograms** | coarse per-characteristic distribution | feeds `S` |

**What this affects beyond the maths:** this list *is* the privacy boundary.
These aggregates are the only things that leave a client. Per-row masks,
features, labels and per-example predictions never do. A new score that needs
something not on this list is a privacy decision, not a formatting one.

## 3.3 The scores — one number per (receiver, donor)

All are in [0, 1]. **Direction matters and is not cosmetic.**

| Score | Formula | Reads as | Directed? |
|---|---|---|---|
| **`W_H`** | Σ H_i·J_j / Σ H_i | "the donor observes together the pairs the receiver frequently lacks together" | directed i ← j |
| **`W_C`** | as W_H, weighted by max(C_i, 0) | as W_H but concentrated on *associated* gaps | directed |
| **`W_marg`** | Σ r_i(1−r_j) / Σ r_i | "the donor observes the features the receiver lacks" — **marginal only** | directed |
| **`s`** | ½(1 + cos(C_i, C_j)) | "these two clients have similar missingness *structure*" | symmetric |
| **`rate`** | ½(1 + cos(r_i, r_j)) | "these two clients have similar missingness *amounts*" | symmetric |
| **`S`** | mean over shared characteristics of Σ_b min(h_i, h_j) | histogram overlap: "these two clients have similar *populations*" | symmetric |
| **`Q`** | (1−λ)·W_H + λ·S, λ = 0.5 | coverage and population combined | directed |
| **`Q_marg`** | (1−λ)·W_marg + λ·S | the marginal-only counterpart of Q | directed |

**The comparisons each score exists to enable:**

- **`W_marg` vs `W_H`** — the same functional form on marginal vs joint
  information. This is the **matched comparator** that isolates what pairwise
  coverage adds. E5's answer: nothing detectable.
- **`s` vs `rate`** — *structure* similarity vs *amount* similarity. These are
  different methods and must never be conflated. E1 found `rate` alone already
  recovers the injected groups perfectly (ARI 1.0).
- **`Q` vs `W_H`** — what population similarity adds. E5: nothing detectable.
- **`Q` vs `S`** — what coverage adds. E5: nothing detectable.
- **`Q` vs `Q_marg`** — what *pairwise* adds beyond marginal + population. E5:
  nothing detectable.

**Critically: a coverage score measures availability, not usefulness.** `W_H`
says the donor observed the pairs the receiver lacks. It does *not* establish
that the donor's model holds transferable knowledge about them. That step is the
hypothesis under test, and the diagnostic (§2.7) is where it fails.

## 3.4 The weighting parameters

| Parameter | Formula position | What it affects | Current value and effect |
|---|---|---|---|
| **α** | base_j = α·q_j + (1−α)·p_j | how much the score is trusted against plain sample size | **α = 1**: score only, sample size does not enter |
| **β** | base_j^β | sharpening — widens the gap between strong and weak donors | **β = 1**: no sharpening |
| **γ** | θ_new = γ·θ_i + (1−γ)·Σ w_j θ_j | how much of its **own** model the receiver keeps | **γ = 0.5**: half. This is what makes it *personalised* rather than one global model |
| **λ_pop** | Q = (1−λ)W + λS | coverage vs population in the combined score | **λ = 0.5** |
| **p_j** | the size anchor | the donor's share of all clients' training rows | used by `fedavg` and by the declared fallback |

**Two things worth saying to a reviewer.**

First, **β = 1 is load-bearing for the negative result.** The geometry analysis
(§2.3, result 4) showed score-weighted mixtures land 0.00–0.34 from the uniform
mixture. With β = 1 the scores simply do not spread donors far enough apart to
change the outcome. A larger β would spread them — but choosing β *after* seeing
that the result was null would be tuning toward a win, so it was not done. The
honest statement is: *at the declared configuration, these scores do not matter*.

Second, **γ is not a nuisance parameter.** `fedavg` uses γ = p_i by definition
(its own sample share, 0.25 with four equal clients) while every other arm uses
γ = 0.5. That is not an inconsistency to fix — changing FedAvg's self-weight
would make it a different algorithm, and then it is no longer the baseline
anyone recognises.

## 3.5 The declared fallback

If a score is **undefined for any donor**, or if **all donors tie within
1e-9**, the score cannot rank donors. The arm then falls back to sample-size
weighting (α = 0, β = 1) and **records that it did**.

**What it affects:** it converts a silent degenerate case into a recorded one.
In E1 it never fired — which is itself worth reporting, because it means no
result depended on it.

## 3.6 The evaluation metrics

| Metric | Definition | What it affects |
|---|---|---|
| **RMSE** | √(mean squared error), original target units | **the only metric that drives a decision.** §8.1 declared it as the effect unit |
| **MAE** | mean absolute error | reported per receiver; never contrasted. A sanity companion |
| **AUC** | ranking quality against y > the client's own training-fold median | reported; never contrasted. Catches "predictions are ordered correctly but badly scaled" |
| **`pos_rate`** | fraction above that median | context for reading AUC |

**Two timepoints, and they answer different questions:**

- **t1** — the aggregated model *as received*, before any local adaptation.
  Measures what aggregation did on its own.
- **t2** — after the receiver adapts it on its own training rows. Measures what
  survives adaptation.

This distinction carries real information: **adaptation compresses differences
between donors.** At t1 the random-donor penalty is +1.7 to +2.9 % on concrete
and wine; by t2 it is under +0.9 %. A few local steps wash out much of what the
weighting did — another reason the scores have little room to matter.

## 3.7 The statistics — how a number becomes a decision

```
Delta% = 100 * (RMSE_arm - RMSE_comparator) / RMSE_comparator      (negative = arm better)
```

1. Compute Δ% per (seed, receiver).
2. **Average over receivers within a seed.** Receivers share data, donors and
   θ₀ — they are not independent.
3. Take the mean over seeds with a **95 % t-interval** (df = seeds − 1).
4. Decide against δ = 2 %:
   - **meaningful** — the whole interval lies beyond ±δ
   - **negligible** — the whole interval lies inside ±δ
   - **unresolved** — otherwise

**What each choice affects:**

| Choice | What would change without it |
|---|---|
| Averaging within a seed first | the effective n would be 40 instead of 10, and intervals would be roughly half as wide — many "negligible" results would become "meaningful" on noise |
| δ fixed in advance | "no difference" could not be distinguished from "not enough power" |
| "unresolved" as a distinct verdict | wide intervals would be read as null results |
| Never pooling datasets | wine's large gain would dominate and hide kin8nm's harm |

**"Negligible" is the load-bearing verdict in this project.** It does not mean
"we found nothing". It means: *the effect, if any, is smaller than the margin we
declared worth caring about, and we had the power to say so.*

---

# 4. The refactored architecture

## 4.1 What the refactor was for

The research code worked but had grown one function — `loop.run_seed`, 119 lines
— that did client construction, local training, signature computation, pairwise
scoring, arm weighting, aggregation, adaptation, evaluation at multiple folds and
budgets, geometry logging, compute accounting and output shaping. Every new
experiment was added by threading another flag through it.

The refactor had one hard constraint: **preserve behaviour exactly**. A golden
reference was captured from the original implementation *before* extraction, and
the equivalence tests compare against it — assignments, masks, signatures, scores
including their `None` decisions, weights, γ, fallback reasons, and a full
end-to-end run.

## 4.2 The component map

```
   experiments/  run_e1   run_e1m   run_e5   pilots   check_*     (unchanged CLIs)
                        |
                        v
   +----------------------------------------------------------------+
   |                      ServerCoordinator                         |
   |  opens rounds, validates updates, aggregates, closes rounds    |
   |  holds NO features, labels or per-row masks                    |
   +------------------+---------------------------+-----------------+
                      |                           |
              +-------v--------+         +--------v---------+
              |   Transport    |         |   Checkpoint     |
              | in-process     |         | generation-based |
              | or mutual TLS  |         | atomic commit    |
              +-------+--------+         +------------------+
                      |
              +-------v---------+
              |  ClientRuntime  |   one per client, one OS process
              |  reads ONLY its own shard                          |
              +-------+---------+
                      |
   +------------------+-----------------+------------------+
   |                  |                 |                  |
+--v-------------+ +--v------------+ +--v-------------+ +--v-------------+
| SignatureProv. | | ScoringStrat. | | WeightingPol.  | | LocalLearner   |
| summarize /    | | summaries ->  | | blend/sharpen/ | | train/adapt/   |
| aggregate_only | | one score     | | normalise/     | | predict/       |
| (the privacy   | | (a registry)  | | fallback       | | evaluate       |
|  boundary)     | +---------------+ +-------+--------+ +----------------+
+----------------+                           |
                                     +-------v--------+
                                     |   Aggregator   |
                                     | states + w     |
                                     | -> new state   |
                                     +----------------+
```

Supporting components: `Registry` (scores, weighting, learners, injectors),
`config` (schema-validated), `seeds` (three independent streams),
`provenance` (revision, dirty flag, dataset checksums, split and mask ids,
preprocessing identity and where it was fitted), `ArtifactStore` (atomic writes,
refuses to overwrite a completed run).

## 4.3 The interfaces, and why each boundary is where it is

| Component | Contract | Why it is separate |
|---|---|---|
| **DataProvider / SplitPolicy / Injector** | load, split, inject | a client with *naturally* incomplete data can skip injection entirely. Previously `build_clients` fused four policies and made that impossible |
| **SignatureProvider** | `summarize` → everything; `aggregate_only` → what may be transmitted | this is the **privacy boundary in code**. Whatever `aggregate_only` returns is what leaves a client; adding a field to it is a privacy decision |
| **ScoringStrategy** | (receiver summary, donor summary) → float or `None` | a new score is one function plus a registration line. `None` is a first-class answer, handled by the declared fallback |
| **WeightingPolicy** | (scores, sizes, receiver) → (w, γ, fallback reason) | scoring and weighting were conflated in one function; the brief required them apart. `w[receiver] = 0` and Σw = 1 are enforced by the aggregator |
| **Aggregator** | (states, i, w, γ) → new state | validates parameter-schema compatibility, refuses a non-zero receiver weight and refuses weights that do not sum to 1 |
| **LocalLearner** | `initial_state`, `train`, `predict`, `metrics`, `evaluate`, `schema` | the optimizer is deliberately **not** persisted: the learner builds a fresh Adam each call, and persisting it would change the algorithm |
| **ClientRuntime** | one client's local work plus the permitted messages | reads only its own shard and refuses a shard belonging to another client |
| **ServerCoordinator** | rounds, validation, aggregation | holds no client data at all |
| **Transport** | in-process or mutual TLS, **one logical protocol** | an in-process run and a networked run take identical code paths through validation and aggregation — only the route differs |

## 4.4 The protocol

Every message is a typed envelope: `protocol_version`, `experiment_id`,
`client_id`, `round_id`, `parent_version`, `schema_id`, `payload_type`,
`update_id`, plus a JSON header and an `.npz` body loaded with
`allow_pickle=False` so an upload cannot carry executable objects.

The coordinator rejects, with a named reason:

| Reason | Meaning |
|---|---|
| `protocol_version_mismatch`, `unknown_experiment`, `unknown_client` | basic validation |
| `identity_mismatch` | the body's `client_id` disagrees with the authenticated identity — the body is never trusted alone |
| `stale_or_late_round`, `parent_version_mismatch` | the update is for the wrong round, or was built on a model the client was not given |
| `schema_mismatch` | the parameter shapes do not match |
| `duplicate_update` | the **same** update arrived twice — benign, e.g. a retry after an interruption |
| `client_already_reported_this_round` | a **different** update from a client that already reported. One contribution per client per round |
| `preprocessing_mismatch` | the client standardises in different coordinates from its peers |
| `malformed_payload` | oversized, malformed or non-finite |

Authentication is the peer's TLS certificate Common Name; authorisation is a
separate `Allowlist` binding an identity to the experiments it may join.
Authentication says *who*, the allowlist says *may they join this one*.

## 4.5 The defects review found, and what they changed

Three framework defects and four dashboard defects were reported in review. Each
was **reproduced before being fixed**, and each now has a regression test that
fails without the fix.

| Defect | Why it mattered | What changed |
|---|---|---|
| **Every client fitted its own feature and target scale** | parameters being averaged represented different numerical coordinates — meaningless, not merely noisy. It also did not match the corrected research protocol | `shared_coordinates()` fits one frozen set; `y_median` stays per client for AUC only, exactly as `loop.scaler_for` does. Every update declares a `preprocessing_id` and the coordinator **refuses to average across a disagreement** |
| **A second, different update from the same client was accepted**, silently replacing the first | one client could choose which of its models was aggregated after watching the round progress | `client_already_reported_this_round`. An identical resend is still `duplicate_update`, because a retry is benign and a different model is not |
| **An interrupted checkpoint replacement destroyed the previous checkpoint** | the recovery mechanism could leave nothing to recover from | generation-based replacement: write `*_g<n+1>.npz`, commit by atomically replacing the header, then purge the superseded generation |
| **A 401 was sent without draining the request body** | on a keep-alive connection the unread body is parsed as the next request; the client could be reset before reading the refusal | drain before refusing; close the connection on an oversized body |
| Dashboard: the explanation showed `p_j` where `base_j` was labelled | the final weights were right, but the explanation misdescribed the method — in the page that exists to explain it | every step displays the quantity its own formula names, with the FedAvg / uniform / fallback special cases spelled out |
| Dashboard: replay used first-seed client sizes for every seed | recomputed size-dependent weights could be wrong for another seed | decided **per arm**: score-only arms are exact for any seed; size-dependent arms are flagged |
| Dashboard: "computed" was animated as "sent" | the picture claimed a delivery the server had not observed | four separate stages; receipt is never claimed because nothing acknowledges it |
| Dashboard: no live prediction error | the demonstration could not show how errors are computed | clients report aggregate error; RMSE = √(sse / n) |

## 4.6 What is verified, and what is not

**Verified** (`docs/VALIDATION.md`): behavioural equivalence with the original
research code against a golden reference; reproducibility including bit-identical
legacy seed derivation; separate client processes completing rounds over mutual
TLS; in-process and networked transports producing identical aggregation to
1e-12; stale, duplicate, already-reported, oversized, malformed, non-finite and
unauthorised updates all refused **before** aggregation; uploads that cannot
execute code; logs containing no payload values or key material; interrupted
rounds resuming without applying an update twice.

**Explicitly not provided** (`docs/SECURITY.md` §3): no secure aggregation, no
differential privacy, no Byzantine robustness. The development certificates are
development certificates. The dashboard has **no authentication** and binds
localhost by default.

---

# 5. The dashboard

`docs/DASHBOARD.md` is the full reference. This is the tour.

## 5.1 What it is, and what it deliberately is not

It is a **presentation layer over the framework**. It trains nothing, scores
nothing and computes no statistics of its own. Every number it shows is produced
by the research code; the browser formats them.

Three guarantees shape it:

1. **Nothing is invented.** Where a number was never saved, it says so rather
   than approximating.
2. **Animation follows observation.** In a live run a packet moves because the
   backend recorded a real handled request. An idle federation is drawn idle.
3. **A single run is never presented as a finding.**

## 5.2 The screens

| Tab | What it shows |
|---|---|
| **Federation** | clients around the coordinator; line thickness is each donor's effective contribution to the selected receiver. Clicking a client shows its per-feature missing rates and, explicitly, what it transmits and what never leaves it |
| **Server aggregation** | the whole weighting calculation in seven steps with this run's real numbers, ending in the balance line γ + Σ donor contributions = 1. A glossary defines every term |
| **Weights** | the receiver × contributor matrix — diagonal is the self-weight, every row sums to 1, every row is different — plus how weights vary across seeds |
| **Results** | per-receiver RMSE, and each arm contrasted against a reference with a 95 % interval and the ±2 % decision. Plus "How was this result computed?" tracing one number back through the pipeline |
| **Timeline & log** | the protocol's fixed order, with an explicit statement that historical runs saved no timestamps, and a replay that steps through it |
| **Live / launch** | a real federation running: process ids, a `https://` coordinator, real events with real timestamps, realised weights, and client-reported prediction error |

## 5.3 The aggregation inspector — the centre of the demonstration

Seven steps, with this run's actual numbers:

1. the receiver-to-donor score for each donor
2. the blend with sample size — showing **base_j**, with p_j alongside
3. sharpening
4. normalisation across donors
5. the receiver's self-weight γ
6. the effective contributions (1 − γ)·w_j
7. the parameter mix

Then the balance line: **γ + Σ donor contributions = 1**, exactly. That is the
check that the result is a proper weighted average.

**The most effective thing to do live** is switch the method selector between
`combined-Q`, `fedavg`, `uniform-donor` and `local-only` and watch the steps
change: for `uniform-donor` the first three steps grey out because it uses
neither a score nor sample size; for `fedavg` step 2 explains that α is pinned to
0 and γ = p_i **by definition**, not by configuration; for `local-only` the whole
thing collapses to one step. That is the clearest possible demonstration that
**scoring, weighting and aggregation are three separate operations**.

## 5.4 What the dashboard refuses to show

| Not shown | Why |
|---|---|
| an event log for historical runs | research runs recorded no events and no timestamps. The timeline is the protocol's fixed order and says so; the replay is labelled a reconstruction |
| squared-error sum and evaluated-record count for historical runs | only the final RMSE was saved |
| training loss curves | never saved |
| validation/test row counts | only `n_train` per client was saved |
| multiple federated rounds in a research run | a research run is a single aggregation; `budget` values are *local adaptation* steps, not rounds |
| confirmed receipt of a delivered model | nothing in the protocol acknowledges a parent fetch, so it is never claimed |
| per-seed client sizes | saved for the first seed only; the inspector flags the arms where that actually changes a weight |

## 5.5 The live run

A launched run is the real system: the framework's `ServerCoordinator`, clients
as **separate OS processes**, **mutual TLS** with per-client certificates, and an
`Allowlist`. Weighting goes through the research code.

Events are separated by what was actually observed:

| Event | Meaning | Shown as |
|---|---|---|
| `computed` | the server aggregated a model | a pulse on the coordinator — **not** a transfer |
| `delivered` | a client requested its model and the server sent the response (HTTP 200) | a packet |
| `transfer` | an update arrived and was accepted | a packet |
| `reject` | an update or request was refused, with the reason | logged, not animated |
| `fallback` | the declared fallback fired | logged |
| `evaluation` | a client reported aggregate error | logged, and fills the RMSE table |

Clients evaluate on their **own** held-out fold and report two numbers per
model — the evaluated-record count and the summed squared error — from which the
dashboard computes

```
RMSE = sqrt(summed squared error / evaluated-record count)
```

which is exactly what `model.metrics` reduces to. No label and no per-example
prediction is transmitted. At round 1 the "received" model is the untrained
starting model, so its error is expected to be large — the page says so.

Launching is frozen: no command field, no path field, no shell. Five validated
options within fixed ranges, unknown keys dropped, `--allow-launch` required.

## 5.6 Suggested ten-minute walkthrough

1. **Federation.** Four clients, each holding its own rows, incomplete in
   *different columns*. The server in the middle holds no records.
2. **Click a client.** Per-feature missing rates, then the explicit statement of
   what it transmits and what never leaves it.
3. **Replay this round.** Signatures up, parameters up, one personalised model
   back per receiver — not one global model. Point at the banner saying this is
   the protocol's order, not a recorded timeline.
4. **Server aggregation.** Walk the seven steps. Stop on the balance line.
5. **Switch the method** to `fedavg`, then `uniform-donor`, then `local-only`,
   and watch the explanation change (§5.3).
6. **Weights.** Every row sums to 1, the diagonal is the self-weight, every row
   is different. That difference *is* the personalisation.
7. **Results.** Per-receiver RMSE, then the contrast table. Say the finding
   plainly: the scored methods sit inside ±2 % of equal donor weighting. The
   threshold was fixed before the experiments ran.
8. **"How was this result computed?"** Follow one number back through
   adaptation, aggregation, weighting and local training — and show the box
   listing what was never saved.
9. **Live / launch.** Start a 2-round, 3-client run. Point at the process ids and
   the `https://` URL — separate OS processes over mutual TLS. Watch real events
   arrive with real timestamps, then the weights and the RMSE table fill in.
10. **Close on the note at the bottom of that tab:** one live run demonstrates
    that the system works; it is never evidence about which method is better.
