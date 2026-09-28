# MechFedGNN — implementation plan (from scratch)

Personalised federated learning for tabular data where clients are incomplete in
**different ways**. Each client summarises its own missingness mask and its own
population; a server turns those summaries into **directed, personalised
aggregation weights**. This round tests whether those summaries predict useful
collaboration.

Write everything yourself — there is no existing code to reuse. Plain **PyTorch
+ NumPy + pandas**. No framework, no abstraction layers, no GNN. One file per
concern; if a file passes 250 lines it is doing too much.

**Stop at the checkpoint at the end of each stage** (§12) and show the user the
output before continuing.

Mask convention everywhere: **`M = 1` observed, `M = 0` missing**; absence
`A = 1 − M`. The mask itself is never missing.

---

## 1. Locked decisions

| Item | Decision |
|---|---|
| Data | GRAPE-9 UCI suite, fetched by a script you write (§2). Candidates **concrete, wine, kin8nm, protein**; a dataset runs only if it passes every §3.5 precondition (at K = 4 all four pass; energy, power and naval are rejected). Complete data + injected missingness. |
| Task | Regression (MSE). AUC-ROC additionally reported by binarising the target at the **training-fold median** and ranking the regression predictions. No second model. |
| Backbone | Mask-aware MLP: input `concat([x_zero_filled, M])`, hidden (64, 32), ReLU, 1 output. |
| Loss | **`L = L_pred` only.** No reconstruction term, no reconstruction head. |
| Clients | `K = 4` in `G = 2` latent groups, 2 clients per group (§4). **Every client acts as receiver in turn** — gives per-client and worst-client results for free. (Was 6; reduced so concrete meets the ≥100 training-row precondition.) |
| Splits | train / val / test = 60/20/20 per client, split first. Summaries use **training rows only**. Tuning on val. Test touched once. |
| Standardisation | Receiver's own training-fold statistics. Never pooled, never from test. Features: **median / IQR over observed entries, no clipping** (IQR 0 → std → 1), so heavy tails are preserved but one extreme value cannot set the scale. Target: mean / std. (A ±5 clip was used in Stage 2 and removed.) |
| Shared init | One `θ0` per run, copied by every client. |
| Seeds | 10 paired seeds: `[11,23,37,53,71,89,101,113,131,149]`. Deterministic torch. |
| β | Fixed at 1. No sharpening this round. |

## 2. `data/download_uci.py` — write this first

Fetch the GRAPE-9 suite from the official GRAPE repository, which ships each
dataset as `data.txt` plus `index_features.txt` and `index_target.txt`. Using
GRAPE's own copies rather than re-deriving from the UCI archive guarantees the
feature/target split matches the GRAPE paper.

- Base URL: `https://raw.githubusercontent.com/maxiaoba/GRAPE/master/uci/raw_data`
- Per dataset `<name>`, fetch `<BASE>/<name>/data/data.txt`,
  `.../index_features.txt`, `.../index_target.txt`
- `data.txt` is whitespace-delimited numeric; the two index files give the
  column indices for features and target
- Datasets: `concrete, energy, housing, kin8nm, naval, power, protein, wine, yacht`
- Write `raw/<name>.csv` with feature columns `f0..f{d-1}` and a final column
  named `target`
- **Assert zero NaNs** in every downloaded file and print `n`, `d` per dataset.
  Controlled injection is only meaningful on complete data — if a file has
  native NaNs, fail loudly.
- CLI: `--out ./raw`, `--datasets concrete wine energy kin8nm power`
- At load time (`data/design.py: load_dataset`), **exact-duplicate feature
  columns are dropped** (first kept), logged, and the reduced `d` reported.
  A duplicate would otherwise be a panel member with φ ≡ its twin.

## 3. `data/inject.py` — panel-based missingness

This is the core of the setup. The masks are **controlled panel-based masks**:
a synthetic design, not a model of any real dataset — no real dataset has been
checked against them. The design borrows the lab-panel idea: a panel is either
ordered for a row or it is not, and when it is not, every feature in that panel
disappears together.

*What independence does and does not remove.* Independent cell-wise masking
still produces features going missing together: at rate 0.3 per feature, two
features are jointly absent with probability 0.3 × 0.3 = **0.09**. What
independence removes is **excess association** — φ ≈ 0, lift ≈ 1 — not joint
absence. Panel masking adds association on top of that chance level, and that
excess is what the C matrix measures. The three masking conditions that separate
these effects are defined in §3.6.

### 3.0 Design split (`data/design.py`)

A flat **10%** of rows (seed 0), with a **150-row floor that applies only when
10% is smaller** (no upper cap), is **excluded from every client** and used only
for column roles (§3.1), panel assignment (§3.2) and histogram bin edges
(§4.3). Frozen design choices therefore never see a row any client trains,
tunes or tests on. Tested: no design row ever appears in a client.

### 3.1 Column roles (computed once per dataset, frozen in config)

- Compute `|corr(f, target)|` on the design-split rows.
- Features constant on the design rows are listed as
  `constant`: never masked, never a characteristic or driver.
- `maskable` = the `floor(d/2)` non-constant features with the **lowest** absolute
  correlation. Masking the least predictive half keeps the task learnable so
  headroom can exist.
- `always_observed` = the complement. These are never masked and serve as the
  **population-similarity characteristics** and the MAR driver set.

### 3.2 Panels

Group the maskable features into panels of related features — panels in real
data measure related quantities, so panel members correlate:

- Compute the `|corr|` matrix among maskable features.
- Greedily form panels of size 2–3: take the unassigned pair with the highest
  `|corr|`, add a third feature if its mean `|corr|` to the pair exceeds the
  median, close the panel, repeat.
- Leftover single features become size-1 panels (they will show no
  co-missingness — that is correct and useful as a contrast).
- Save the panel assignment to the config and print it. It is **identical for
  every client** — clients differ in how often they order each panel, not in
  which features belong to a panel.

### 3.3 Row-level ordering decision

For client `k`, row `r`, panel `P`:

```
ordered(r, P) ~ Bernoulli( q_k(r, P) )
if not ordered:  every feature in P is missing for row r
if ordered:      each feature in P is independently missing with prob `jitter`
```

`jitter` default **0.05** — occasional single-feature loss inside an ordered
panel. Its effect on φ is arithmetic, not a realism claim: with `jitter = 0`
every panel member has an identical mask column and within-panel φ is exactly
**1.00**; with `jitter = 0.05` it is **≈ 0.82–0.84** on every dataset here
(§3.5 validation).

The mechanism decides `q_k(r, P)`:

| Mechanism | `q_k(r, P)` | Story |
|---|---|---|
| `mcar` | `p_k[P]`, constant | the panel is skipped at random |
| `mar` | `sigmoid(a + b · z_r)`, `z_r` = standardised linear score over the **always-observed** features | ordering depends on things we can see |
| `fd_mnar` | rank rows by the panel's own mean value; skip the top (or bottom) `1 − p_k[P]` fraction | the panel is skipped when its own values would have been extreme |
| `cd_mnar` | `p_k[P]` shifted by the row's target quantile bin | ordering depends on the outcome |

For `mar`, `fd_mnar` and `cd_mnar`, **calibrate `a` (or the threshold) so the
realised ordering rate equals `p_k[P]`**. Without calibration, mechanism gets
confounded with missing rate, which is fatal for the one comparison that is
supposed to isolate mechanism.

### 3.4 Knobs

- `p_k[P]` — per-client, per-panel ordering probability. This is what makes
  clients differ. Set by the client profile (§4).
- `jitter` — within-panel independent loss, default 0.05.
- `driver_overlap = o ∈ [0,1]` (MAR only) — the correlation between panel
  drivers. The always-observed features are whitened and `n_panels + 1`
  orthonormal directions drawn: `u_0` shared, `u_P` panel P's own. Panel P's
  driver is

  ```
  z_P = sqrt(o) · z_shared + sqrt(1 − o) · z_own(P)
  ```

  so any two panel drivers correlate exactly `o` in-sample. At 0 the drivers
  are uncorrelated and panels are skipped independently; at 1 all panels are
  skipped on the same rows, so co-missingness spans panels as well as within
  them. **This is the dose-response knob.** Requires `n_panels + 1 ≤` number
  of always-observed features (asserted).

  *Why not "the fraction of panels sharing one driver"* (the original
  wording): with 2 panels — concrete, wine, energy — that fraction can only
  be 0, ½ or 1, and ½ means one panel shares a driver with nobody, so
  `o = 0.5` would equal `o = 0` and E1 would have no cross-panel structure.
  The continuous form gives a smooth dose-response on any panel count.
- `class_spread ∈ [0,1]` (CD-MNAR only) — how unequal the per-outcome-bin rates
  are. At 0 it collapses to MCAR.

Both knobs at 0 must return the setup to **independent ordering decisions
across panels** at the same marginal rates — cross-panel φ ≈ 0. That is the
null condition for the cross-panel knobs; verify it. It is **not** an
independence control: within-panel dependence remains by construction at any
knob setting. The independence control is condition (a) in §3.6.

### 3.5 Diagnostics the injector must report (`data/validate_injector.py`)

For each configuration, over several seeds:

- realised overall missing rate, and per-feature rates
- **within-panel mean φ** — should be high (≈ 0.7–0.95 with jitter 0.05)
- **cross-panel mean φ** — should be ≈ 0 under MCAR and rise with
  `driver_overlap` / `class_spread`
- lift matrix mean and max
- a **missing-panels-per-row** table: fraction of rows with 0, 1, 2, … missing
  panels. If almost every row is either complete or missing everything
  (> 90%), flag it **ALL-OR-NOTHING**: the panels then act as one block and
  there is no panel structure left for clients to differ in. (This is a
  structural check, not a realism check.)

Two hard checks, printed with a PASS/FAIL verdict:

- **Null:** under MCAR with `driver_overlap = 0`, cross-panel φ must be within
  sampling noise of 0. Use the analytic SE `sqrt((1-p²)/(n·p²))` rather than a
  fixed tolerance — a fixed tolerance just measures `n`.
- **Dose-response:** sweeping `driver_overlap` 0 → 1 must raise cross-panel φ
  monotonically. Same for `class_spread`.

**Dataset precondition**, printed as ACCEPT/REJECT: reject a dataset if the
median `|corr|` among its maskable features (design rows) exceeds **0.9**, or if
any mechanism is flagged ALL-OR-NOTHING, or if it forms **fewer than 2 panels**
(both latent groups would get the same profile, §4.1), or if **any client has
fewer than 100 training rows** (under 120 prints a borderline warning; also
enforced in `build_clients`). Near-collinear maskable features make panels
copies of one signal. Rejected so far (at K = 4): naval (median 0.995),
energy (92 training rows/client), power (d = 4 -> one panel). concrete (88 at
K = 6) passes at K = 4 with 132. **Headroom (§8) is not a precondition and
never excludes a dataset.**

Save the output to `results/injector_validation/<dataset>.txt`.

### 3.6 The three masking conditions

Per-feature missing rates are matched **in expectation** across all three:
feature `f` in panel `P` is missing with probability `1 − p[P]·(1 − jitter)`.

| Condition | Mechanism | Within-panel association | Cross-panel association |
|---|---|---|---|
| **(a) independent cell masking** | `cell`: every maskable cell independently missing at its feature's expected rate | none (φ ≈ 0) | none (φ ≈ 0) |
| **(b) independent panel masking** | `mcar`: panels ordered independently | yes, by construction | none (φ ≈ 0) |
| **(c) coupled panel masking** | `mar` with `driver_overlap > 0` (E1: 0.5) | yes | yes, rising with the knob |

(a) is the only **independence control**; (b) is what "cross-panel knob = 0"
gives, and it still carries within-panel dependence. Validation reports, per
condition and feature, the **expected** rate (the formula above) and the
**realised** rates (mean and range over seeds, plus one seed's raw counts):
expected rates match exactly by construction, realised ones differ by sampling.

## 4. `data/clients.py` — how clients are constructed

Two things vary across clients, and they must vary **independently**: which
panels a client habitually skips (missingness structure), and which patients it
sees (population). An earlier attempt entangled them and the experiment became
uninterpretable.

### 4.1 Latent missingness groups — give the scores something real to find

Define `G = 2` groups of clients. Each group has a **panel profile**: a vector
of ordering probabilities over panels.

- Split the ordered panel list in half. Group A rarely orders the **first
  half** (`p = 0.2`), the rest usually (`p = 0.9`)
- Group B is the mirror image — rarely orders the **second half**, the rest usual
- With an odd panel count the **middle panel** stays usually-ordered for both
  groups. It is a within-design control: no score should favour either group
  on it
- Each client's profile = its group profile + small per-client noise
  (`±0.05`, clipped to [0.05, 0.95]), so clients within a group are similar but
  not identical

`K = 4` clients: 2 per group; the halves rule above is unchanged. This is the **ground truth**. Two clients in the
same group lose the same panels; two clients in different groups lose
complementary panels — so a donor from the *other* group is exactly the one that
observes what the receiver lacks. Without this latent structure there is nothing
for any score to detect, and a null result would be uninformative.

Record the true group labels in the config. They are used **only for
evaluation**, never as an input to any score.

### 4.2 Row assignment (population)

Two modes, selected by config:

- `population_homogeneous` (E1, E2): rows shuffled and split into `K` disjoint
  equal parts. Populations match; only missingness differs.
- `population_stratified` (E3): pick the `always_observed` feature with the
  highest `|corr|` with the target as `partition_col`; split rows at its median
  into LOW and HIGH pools; each client draws with its own `frac_low`. Assign
  `frac_low` so that the **population grouping cross-cuts the missingness
  grouping** — i.e. each missingness group contains both low-population and
  high-population clients. That is what creates a genuine conflict between
  coverage and population similarity.

Rows are disjoint across clients in both modes, and drawn only from the
non-design pool (§3.0).

### 4.3 Population-similarity characteristics

Per client, normalised histograms of every `always_observed` feature, on shared
bin edges = **deciles over the design-split rows** (§3.0), computed once and frozen
in the config.

### 4.4 Sizes

Every client keeps its full, equal share — the receiver included. The
receiver instead differs by **missingness asymmetry**: while it is the
receiver, a client orders the panels its group skips at `receiver_p_rare`
(default **0.15**) instead of the group baseline `p_rare = 0.2` (applied as a
shift of its own noisy profile), and its mask is re-injected with the same
random draws, so only those panels change. Size, rows and splits are
untouched.

*Why not `receiver_fraction` (the original knob):* shrinking the receiver to
0.2–0.3 of its size left it 7–29 training rows; headroom then mostly measured
"a 7-row model is bad", adaptation overfit, and the single-seed pilot was
non-monotone (Stage 2 commit `70e3408`).

## 5. Quantities (`signatures.py`, pure NumPy)

From a client's training mask `M` of shape `(n, d)`:

```
r[f]    = P(A_f = 1)                                            per-feature rate
H[f,g]  = P(A_f = 1, A_g = 1)                                   joint absence
J[f,g]  = P(M_f = 1, M_g = 1)                                   joint observation
C[f,g]  = (H[f,g] − r_f·r_g) / sqrt(r_f(1−r_f)·r_g(1−r_g))      phi
```

- `C` range [−1, 1], diagonal 0. If a feature is always observed or always
  missing, φ is undefined → set its row/col to 0 and keep `r` separately.
- Return `chi2 = n · C²` (φ² = χ²/n) as a significance check.
- Test the identity `J = 1 − r_f − r_g + H`.
- **These three are different things and must never be conflated:** `H` is how
  often a gap happens, `C` is how associated two gaps are, `J` is how often a
  sender sees the pair.

Population similarity:
```
S(i,j) = mean over shared characteristics of  Σ_b min(h_i[u][b], h_j[u][b])
```
Histograms normalised, shared bins. A characteristic missing from either client
is **dropped from the mean, never scored 0**. No shared characteristics → `None`.

## 6. Scores (`scores.py`)

```
W_H(i←j) = Σ_{f<g} H_i[f,g]·J_j[f,g]  /  Σ_{f<g} H_i[f,g]
W_C(i←j) = Σ_{f<g} max(C_i[f,g],0)·J_j[f,g] / Σ_{f<g} max(C_i[f,g],0)
s(i,j)   = 0.5·(1 + cosine(vec upper C_i, vec upper C_j))     missingness similarity
rate(i,j)= 0.5·(1 + cosine(r_i, r_j))                         marginal-rate similarity
Q(i←j)   = (1 − λ_pop)·W_H(i←j) + λ_pop·S(i,j)
```

All in [0, 1]. `W_H`, `W_C` are **directed**; `s`, `rate`, `S` are symmetric.
Denominator 0 → return `(None, defined=False)`.

Fallback hierarchy: both available → the formula; `W` undefined → use `S`;
`S` is `None` → use `W`; neither → `None`, kernel falls back to the size anchor.

**Docstring this:** `W_H` means "the sender observes together the pairs the
receiver frequently lacks together." It does NOT measure feature importance and
does NOT establish that the sender's model holds transferable knowledge. That is
the hypothesis under test.

## 7. Aggregation (`kernel.py`)

```
base_j = α·q_j + (1 − α)·p_j        # q_j None → base_j = p_j
w_j    = base_j^β / Σ_{l≠i} base_l^β
θ_new  = γ_i·θ_i + (1 − γ_i)·Σ_{j≠i} w_j·θ_j
```

`p_j` = client sizes normalised over ALL clients. **FedAvg limit, unit-tested to
1e-8:** `α=0, β=1, γ_i=p_i` gives `θ_new = p_i·θ_i + Σ_{j≠i} p_j·θ_j`.
(Not `γ=0` — with two equal clients that swaps their models.)

### 7.1 Declared fallback when a score cannot distinguish donors

For a score arm and receiver `i`, the score provides **no basis for
distinguishing donors** when either (i) the score is undefined for **any**
donor, or (ii) all donors' scores are equal to within `1e-9` (numerical
equality). In both cases the arm falls back, for that receiver-seed, to the
**declared fallback: sample-size weighting** (`w_j = p_j / Σ_{l≠i} p_l`, the
arm's own `γ` kept) and the fallback and its reason are recorded and reported.
Case (i) replaces mixing scores and sizes in one weight vector, which put two
different scales side by side. Equal weighting is **a fallback, not a justified
optimum** — nothing here shows it to be the right weighting when scores tie.
With equal client sizes (this setup) the fallback weights are numerically
equal; the distinction matters for the record and for unequal sizes.

*Hypothesis, not established:* when masks carry no usable signal, neither
coverage nor similarity provides a basis for distinguishing donors, and the
fallback applies. Two caveats are part of the hypothesis:

- **Identical mask signatures do not imply identical learned relationships,
  training quality or sample size.** Two donors with the same `r`, `H`, `J`, `C`
  can hold very different models; every mask-based score is blind to that.
- **Residual structure is not assigned to similarity alone.** Coverage scores
  (`W_H`, `W_C`) may also discriminate through residual structure in the mask
  beyond what similarity captures. Both are measured; neither is presumed.

## 8. Round protocol (`loop.py`) and headroom (`headroom.py`)

1. One `θ0`; every client copies it.
2. Each client trains locally, `L_pred` only (Adam 1e-3, 300 steps, batch 64).
3. Compute §5 summaries from each client's **training rows**.
4. For each receiver and each arm, compute weights (§7).
5. Aggregate; evaluate on the receiver's test fold → **timepoint 1**.
6. Adapt on the receiver's training rows for the frozen budget; evaluate →
   **timepoint 2**.
7. `local-only` gets the same total step count (300 + budget).

Measured transfer benefit, evaluation only, never a weight input:
```
U(i←j) = L_i(local reference) − L_i(model incorporating j)
```

**Headroom is a reference, not a gate.** It is reported *beside* every results
table as context. It must **never suppress an experiment or exclude a
dataset**: a receiver or dataset with thin or no headroom is reported as such
and serves as a control. The pooled model is one particular training
procedure, not an upper bound for personalised aggregation: a model that
weights a subset of donors can beat `local-only` even where indiscriminate
pooling does not, and pooling cannot represent "weight two donors highly and
two at zero" at all.

On the receiver's test fold, under its own mask: `loss_local` (own training
rows only) vs `loss_pooled` (one model trained on the pooled training rows of
all clients, evaluated under the receiver's mask). Both references use the
**same stopping rule** — from `θ0`, receiver's standardisation, Adam,
early-stopped on the receiver's validation fold (check every 25 steps; no stop
before step 100; patience 50 checks = 1250 steps; max 5000 steps, 20000 on
protein, whose references hit 5000; best checkpoint restored). **Identical
stopping criteria do not mean equal compute**: the pooled reference sees more
rows per epoch and usually runs longer, so compute is recorded separately
(below). `headroom = loss_local − loss_pooled`. Verdicts, in this order:

- **POOLING HARMS** — pooled is worse than local *beyond sampling noise*:
  on the receiver's test rows, `d_r = e²_pooled,r − e²_local,r`, and
  `mean(d) > 2 · sd(d)/√n` (paired, same rows).
- **NO HEADROOM** — `headroom/loss_local ≤ 0.02` (includes negative headroom
  within noise).
- **THIN** — `≤ 0.10`. Else **OK**.

Also report the downside explicitly: `pooling_harm = loss_pooled − loss_local`
where positive (0 otherwise), and the number of receiver-seeds in which pooling
is worse than local. Pooling is a diagnostic only — never an arm, never a
weight source.

**Companion diagnostics**, reported beside headroom because the pooled
reference alone cannot show what subset weighting could achieve:

- **Individual donor candidates vs `local-only`** — for every donor `j`, the
  pairwise mix `γ·θ_i + (1 − γ)·θ_j`, adapted with the same budget; its test
  loss minus `local-only` is `U(i←j)` (§8 above), at both timepoints.
- **Validation-selected candidate, evaluated on test** — among `local-only`
  and the single-donor candidates, pick the one with the lowest *validation*
  RMSE (timepoint 2) per receiver-seed, and report its *test* RMSE against
  `local-only` and `fedavg`. Selection never sees test.
- **Score-based aggregation vs FedAvg** — every score arm's paired relative
  difference against `fedavg`, with the §8.1 uncertainty.

**Compute accounting**, recorded per model and reported per arm: optimiser
steps, **examples processed** (`steps × min(batch, n_train)`), and wall-clock
seconds — for local training, each arm's adaptation, the single-donor
candidates, and both headroom references. An arm's examples include the local
training of every client whose parameters it mixes.

### 8.1 Uncertainty and the meaningful-effect threshold (fixed before E1)

- **Effect unit: relative error, in percent of the comparator's RMSE.** For an
  arm `a` against comparator `c`, per receiver-seed:
  `Δ% = 100 · (RMSE_a − RMSE_c) / RMSE_c`. Negative = `a` is better.
  (Percent of the comparator's error — **not** percentage points of anything.)
- **Dependence.** Receivers in the same seed share data, donors and `θ0`, so
  they are not independent. The unit of replication is the **seed**: average
  `Δ%` over receivers within a seed, then take the mean and a 95 %
  t-interval over the seed means (df = seeds − 1). Datasets (scenarios) are
  never pooled; each gets its own interval. Per-receiver effects are reported
  descriptively beside it.
- **Meaningful-effect threshold `δ = 2 %`** relative RMSE, fixed here before
  any E1 output exists. It is the same 2 % already used for NO HEADROOM (§8),
  chosen before any pilot result.
- **Decision rule, per contrast and dataset:** *meaningful* if the whole 95 %
  interval lies beyond `±δ` in one direction; *negligible* if it lies inside
  `(−δ, +δ)`; otherwise *unresolved*. No contrast is called noise because of a
  gap under 1 % or because rankings change between budgets — only the interval
  decides.
- **Can a bounded study resolve `δ`?** From the pilot seeds (validation fold),
  estimate the SD of seed-level `Δ%` per contrast and dataset, and predict the
  interval half-width at the study's seed count:
  `hw = t(0.975, n−1) · SD / √n`. The study can resolve `δ` for a contrast if
  `hw ≤ δ/2` (then a true zero lands inside `±δ` and a true effect of `1.5·δ`
  or more lands outside). Contrasts that fail this are reported as
  underpowered, not dropped.
- **Pilot result, recorded before E1** (`python pilots.py --power`; seeds 11,
  23, 37, validation fold, t2, `adapt_budget = 10`; 21 contrasts per dataset):
  at 10 seeds the predicted half-width is within `δ/2` for **21/21 on wine,
  kin8nm and protein** and **13/21 on concrete** — concrete's contrasts
  against `local-only` (SD 1.7–2.4 % across seeds) are underpowered at this
  study size. A seed-level SD estimated from 3 seeds is itself rough; the
  E1 intervals, not this prediction, are what count.

## 9. Arms (fixed list, always all of them)

```
local-only              no collaboration
fedavg                  α=0, β=1, γ_i=p_i
uniform-donor           equal donor weights, same γ as the score arms
marginal-rate           q = rate(i,j)
missingness-similarity  q = s(i,j)
coverage-W_H            q = W_H(i←j)
coverage-W_C            q = W_C(i←j)
population-S            q = S(i,j)
combined-Q              q = Q(i←j), λ_pop = 0.5
```

`uniform-donor` is essential: without it, a "gain" may just be collaborating at
all rather than weighting well.

## 10. Experiments

Each runs on every dataset passing the §3.5 preconditions, **reported per dataset, never pooled**,
every client as receiver in turn, 10 seeds. Each produces a report via the
`run-report` skill.

**E1 — signal comparison (the main one).**
`population_homogeneous`, MAR with `driver_overlap = 0.5`, rate 0.3, the latent
group structure of §4.1. All nine arms.
**Primary question:** does any weighting rule **avoid the harm that
indiscriminate averaging causes**, on the receivers where `local-only` beats
`fedavg` and `uniform-donor`? Only then: does it beat `local-only`?

*Correction to the record.* An earlier revision framed this as harm from
indiscriminate **pooling**, on the strength of negative headroom in the
synthetic tests and the K = 6 wine pilot. That finding did not survive: with
both references early-stopped under one protocol, the pooled reference is
never worse than local beyond sampling noise — **0 of 48 receiver-seeds**
(POOLING HARMS, §8) across concrete, wine, kin8nm and protein. Pooling (one
model trained on everyone's rows) and parameter averaging (the arms) are
different things; the harm Stage 2 did show was at the **arm** level —
`local-only` beat `fedavg` on kin8nm at every adaptation budget.

E1 must therefore report, **per receiver** (mean over seeds, at both
timepoints): `local-only − fedavg` and `local-only − uniform-donor` (RMSE;
negative = the averaging arm is worse, i.e. harms), and for **every arm** how
many receivers — and receiver-seeds — it harms relative to `local-only`.

Secondary questions: does any score's donor ordering track measured `U`? Does
any arm beat both `local-only` and `uniform-donor`?

**Run protocol (`run_e1.py`).** Every dataset passing §3.5 runs — concrete,
wine, kin8nm, protein — with **none excluded for thin or absent headroom**;
thin-headroom datasets are reported as thin and serve as controls. 10 seeds,
every client as receiver, all nine arms, validation and test folds (validation
only for the selected-candidate diagnostic). Effects are judged by §8.1 against
`δ = 2 %`. The graph extensions (§15) stay deferred.
**Also report group recovery:** cluster clients into 2 groups by each symmetric
score (`rate`, `s`, `S`) and report the Adjusted Rand Index against the true
group labels. `s` should recover the missingness groups; `rate` should do worse
(clients differ in structure, not in overall rate); `S` should be at chance under
`population_homogeneous`.

**E1 outcome (recorded 2026-09-28; runs `results/e1/20260928_f86e996_*`).**
Pre-registered analysis (§8.1, δ = 2 %, seed = replication unit, n = 10):

- **Score-based donor weighting vs the equal-donor control.** All 48
  score-arm vs `uniform-donor` contrasts (6 score arms × 4 datasets × 2
  timepoints) have their **entire 95 % interval inside ±2 %** → negligible.
  Largest interval bound: +1.44 % (concrete, `coverage-W_C`, t1).
- **Differences from `fedavg` are reproduced by the equal-donor control**
  (`uniform-donor`, γ = 0.5, equal donor weights) and **do not establish an
  advantage from score-based donor weighting.** FedAvg's self-weight is
  `p_i` = 0.25 with four equal clients; changing it would make a different
  algorithm, so no matched-γ FedAvg is run — `uniform-donor` is the control
  that isolates donor weighting.
- **Averaging harm (primary question).** kin8nm: every mixing arm harms 3/4
  receivers; `fedavg` vs `local-only` +1.9 % [1.3, 2.4] (inconclusive against
  δ), score arms +1.0–1.3 % (negligible). protein: every mixing arm harms 4/4
  receivers, all contrasts negligible (+0.4–0.8 %). No weighting rule avoids
  the harm; the harm itself is below δ except `fedavg` on kin8nm, whose
  interval straddles it. wine: all mixing arms meaningfully better than
  `local-only` (−6.6 to −7.5 %); concrete: −3.8 to −4.1 %, mostly
  inconclusive (underpowered, as §8.1 predicted).

**Group recovery — limitation, stated plainly.** `rate` (marginal-rate
similarity) achieves **ARI 1.0 on all four datasets**. In this construction
the latent groups differ in their ordering *rates*, so **joint missingness is
unnecessary to identify the injected groups**, and group recovery **cannot
support the claim that pairwise structure adds information beyond marginal
rates.** The §10 prediction ("`s` should recover the missingness groups;
`rate` should do worse") **failed**: `s` reached ARI 1.0 on kin8nm, 0.8 on
wine, 0.3 on protein and −0.05 on concrete, while `rate` was perfect
everywhere.

*Singleton check (exploratory, `analysis/singleton_c_check.py`, 10 seeds).*
On concrete (`{f2}`) and protein (`{f6}`) the cross-panel `C` entries that
involve the single-feature panel are small but **not zero** (mean +0.06 to
+0.10). Group information in `C` sits in the **magnitude** of the within-panel
φ — ≈ 0.94 in the group that rarely orders the panel, ≈ 0.65 in the group that
usually orders it, so even that difference is **rate-driven** here. With two
multi-feature panels (wine, kin8nm) the high/low pattern flips between groups
and cosine similarity separates them (`s` within 0.994–0.998 vs between
0.967). With a singleton there is only one within-panel block; both groups'
`C` vectors point the same way at different lengths, and **cosine similarity
discards length** — between-group `s` 0.9935 vs within 0.9922 (concrete),
0.9991 vs 0.9997 (protein).

*Ranking vs weighting (exploratory, `analysis/e1_ranking_vs_weighting.py`,
test fold, γ = 0.5 and budget 10 matched throughout).* `s` and `S` scores
barely vary across donors (relative range ≤ 0.035), so their weights are
≈ uniform (max weight ≤ 0.340) before any normalisation — the compression is
in the score, not the normaliser. `rate`, `W_H` and `W_C` do produce
non-uniform weights (max weight 0.42–0.56), yet their arms still land within
±0.4 % of `uniform-donor`. The top-ranked single donor is not reliably better
than a uniformly drawn one (all contrasts negligible except concrete t1,
inconclusive); on kin8nm and protein it is often the least-harmful donor
rather than a positive one (`rate`'s top donor is best-but-still-worse than
`local-only` in 40 % / 52 % of receiver-seeds at t2). The oracle-best single
donor beats the uniform mixture by 0.4–3.2 % at t2, but a validation-selected
single donor does not (negligible everywhere at t2). Differences present at
t1 shrink after adaptation (e.g. concrete top-by-`rate` vs uniformly drawn:
−3.3 % → −1.0 %).

**E1M — matched-marginal test (specified, not run; runs before E2).** The
decisive test of the thesis claim. It is the only construction where `rate`
is **flat by design**, so pairwise structure alone can distinguish clients.

- **Masks.** Every client orders every panel with the **same** probability
  `p` (rate 0.3 → `p = 0.7/0.95`), `jitter = 0.05`, **profile noise 0**, and
  **no receiver asymmetry** (it would break flatness). The groups differ
  *only* in **which features go missing together**: group A uses panel
  partition `π_A`, group B uses `π_B`, over the same maskable features.
  Both partitions consist of **pairs** (a shared singleton only when the
  maskable count is odd), and **share no pair**. `π_A` is the design-split
  greedy pairing; `π_B` is the no-shared-pair alternative whose mean
  within-pair `|corr|` on the design rows is closest to `π_A`'s, so feature
  relatedness is balanced between groups.
- **Mechanisms.** Primary: condition (b), independent panel masking.
  Secondary: condition (c), `mar` with `driver_overlap = 0.5`. Null control:
  condition (a), cell masking for every client (no structure to find).
- **Everything else matched to E1:** homogeneous populations, K = 4 (2 per
  group), same seeds, model, γ = 0.5, `adapt_budget = 10`, all nine arms,
  folds, and the §8.1 analysis with δ = 2 %. Datasets: concrete, wine, kin8nm,
  protein (all have ≥ 4 maskable features, so two pair partitions with no
  shared pair exist).
- **Design checks, reported before any arm result is read:** expected
  per-feature rates identical across clients by construction; realised
  per-feature rates within binomial noise (report the max `|r_i − r_j|`);
  `rate` similarity equal within and between groups; `rate` ARI at chance;
  within-panel φ of equal magnitude in both groups (removing E1's rate-driven
  magnitude signal); `s` ARI near 1.
- **Questions, in order.** (1) Ground truth: does transfer depend on
  structure — mean `U(i←j)` for same-partition vs other-partition donors? If
  not, no mask score can help here, and that is the finding. (2) Does any
  pairwise-structure arm (`missingness-similarity`, `coverage-W_H`,
  `coverage-W_C`) beat `uniform-donor` with its whole interval beyond −2 %?
  (3) Does the top-ranked donor by `s` / `W_H` beat a uniformly drawn donor?
  (4) Is `marginal-rate` ≈ `uniform-donor`, as flatness requires?
- **Expected, recorded now:** under matched marginals the within-panel
  `J_j[f,g]` is higher on the donor's *own* pairs, so `W_H` and `s` both
  favour **same-partition** donors — coverage and similarity should rank
  donors the same way here, which removes the coverage/similarity conflict
  but not the question of whether either helps.
- **Needs code:** per-group panel partitions in the injector (clients carry
  their own `panels`); everything downstream already works from masks.

**E2 — mechanism sweep.**
Repeat E1 across `mcar`, `mar` (driver_overlap 0 / 0.5 / 1.0), `fd_mnar`,
`cd_mnar` (class_spread 0 / 0.5 / 1.0), at rates 0.2 / 0.4.
Question: do the pairwise scores help more when missingness has relational
structure (high cross-panel φ) than when it does not? MCAR with
`driver_overlap = 0` is the null — all scores should collapse toward
`uniform-donor` there. Plot/tabulate every score's `U`-agreement against
cross-panel φ.

**E3 — population vs coverage conflict.**
`population_stratified`, assignment chosen so the population grouping cross-cuts
the missingness grouping (§4.2). Sweep `λ_pop ∈ {0, 0.25, 0.5, 0.75, 1}`.
Report `Q_vs_uniform = loss(uniform-donor) − loss(combined-Q)`.
Assert `|Q_i − Q_j| > 0.05` for at least three λ values; if not, the two signals
cancelled and the run is **void** — say so rather than reporting it.

## 11. Repo layout

```
mechfedgnn/
├── configs/            defaults.yaml, e1.yaml, e2.yaml, e3.yaml
├── data/
│   ├── download_uci.py      §2
│   ├── design.py            §3.0
│   ├── inject.py            §3
│   ├── validate_injector.py §3.5
│   └── clients.py           §4
├── signatures.py       §5
├── scores.py           §6
├── kernel.py           §7
├── loop.py             §8
├── headroom.py         §8
├── model.py            mask-aware MLP
├── run_e1.py  run_e2.py  run_e3.py
├── report.py           builds REPORT.md per the run-report skill
└── tests/
```

## 12. Stages — stop after each

**Stage 0 — data and injector.** Write `download_uci.py`, fetch concrete /
housing / wine, write `inject.py` and `validate_injector.py`, run validation on
each dataset.
**STOP** — show: the panel assignment per dataset, the within-panel vs
cross-panel φ table, the null and dose-response verdicts, and the
missing-panels-per-row table.

**Stage 1 — quantities and scores.** Write the §13 tests first (they fail), then
`signatures.py`, `scores.py`, `kernel.py` until green.
**STOP** — show full pytest output.

**Stage 2 — clients, model, loop, headroom, report.** Then pilot the adaptation
budget: scan `{0, 10, 25, 50, 100}` over 3 seeds on the validation fold, pick the
smallest that separates arms stably, freeze it in `defaults.yaml` with a printed
justification. Report headroom per client (test fold, early-stopped references),
client training sizes, and how often the ±5 input clip fires.
**STOP** — show headroom verdicts per client and the chosen budget.

**Stage 3 — experiments.** `run_e1.py`, then `run_e2.py`, then `run_e3.py`.
**STOP after E1** — show the full report before running E2.

## 13. Tests (write these before the code they cover)

Fixture: 3 clients, features `[X, Z, Y]`, 100 rows each. Client A has 50 rows
fully observed and 50 rows with X and Z missing together; Y always observed.

- `C_A[X,Z] == 1.0`; `H_A[X,Z] == 0.5`; Y's pairs 0 by the constant-feature rule
- masks for B and C giving `J_B[X,Z] = 0.9`, `J_C[X,Z] = 0.3`
- `W_H(A←B) == 0.9`, `W_H(A←C) == 0.3` (one positive pair → score equals J)
- kernel with α=1, β=1 over {B,C}: weights `0.75 / 0.25`; with γ_A=0.5,
  `θ_new == 0.5·θ_A + 0.375·θ_B + 0.125·θ_C`
- FedAvg limit, K=3 unequal sizes, tol 1e-8
- `J == 1 − r_f − r_g + H` on random masks
- φ invariance: `C(M) == C(1 − M)`
- **panel injector:** with `jitter = 0`, within-panel φ is exactly 1 and every
  panel member has an identical mask column; with `jitter = 0.05`, within-panel
  φ is high but < 1
- **null:** MCAR with `driver_overlap = 0` gives cross-panel φ within 3·SE of 0
- **dose-response:** cross-panel φ increases monotonically with
  `driver_overlap` and with `class_spread`
- **rate calibration:** all four mechanisms at the same `rate` produce realised
  overall rates within ±0.02 of each other
- **panels identical across clients**, ordering probabilities differ
- `S`: identical histograms → 1.0; disjoint → 0.0; a characteristic in only one
  client is ignored (not 0); none shared → `None`
- fallback hierarchy: all four branches
- headroom detects both states: OK when pooling adds information (small
  clients, no latent groups), NO HEADROOM when the receiver's own sample is
  abundant
- missingness asymmetry changes only the receiver's rare panels; size unchanged
- clients with < 100 training rows are refused
- determinism: same seed → identical CSVs

## 14. Do not build

Any GNN or GRAPE backbone · PyTorch Geometric · reconstruction loss or head ·
attention · learned encoders · sharpening (β ≠ 1) · Ditto, FedProx, Clustered
FL · DARN, Cafe, FedSaC · eICU or MIMIC · multi-round federation · differential
privacy · wandb · any abstraction layer "for later extensibility".

A negative result reported clearly is a valid outcome of this round. Do not tune
toward a win.

## 15. Graph representation (explicit, no learning) — experiment E4

> **Status: specified only. Do not implement or run any of §15 until E1 is
> done and its report has been reviewed.**

Two additional arms that read the receiver's co-missingness as a **graph**
rather than as a list of pairs. Both are closed-form NumPy — no learning, no
new hyperparameters beyond `τ`. They are consistent with §14: nothing here is
a GNN or a learned encoder.

**E4** runs on the **same setup and the same seeds as E1** (same datasets,
clients, masks, `θ0`, budgets), with all nine §9 arms plus the two below, so the
comparison with E1 is exact (the determinism test guarantees the nine E1 arms
reproduce bit-for-bit).

### 15.1 The receiver's co-missingness graph

From receiver `i`'s training mask (§5): nodes = the **maskable** features;
an edge `(f,g)` wherever `[C_i]₊[f,g] > τ`, with **`τ = 0.2`**.

*Permutation-threshold variant* (reported alongside, not tuned): shuffle each
maskable column of the receiver's training mask independently (keeps every
feature's rate, destroys co-missingness), recompute `[C]₊`, and pool the
off-diagonal values over `B = 200` shuffles; `τ_perm` = their 95th percentile.
`B` and the percentile are fixed constants of the variant, not knobs.

Connected components `K_1 … K_c` of this graph. **Every** maskable feature
belongs to exactly one component; an isolated feature is a **single-feature
component** — a legitimate component with an empty edge set, not an error case.
Single-feature panels are real in this setup (concrete `{f2}`, protein `{f6}`),
and a feature can also end up isolated because its co-missingness falls below
`τ` in this receiver's mask.

### 15.2 `component-W_comp`

```
ω_k           = mean of H_i[f,g] over the edges (f,g) of component K_k
A_j(K_k)      = P( M_f = 1 for every f ∈ K_k )      measured directly on donor j's training mask
W_comp(i←j)   = Σ_k ω_k · A_j(K_k)  /  Σ_k ω_k
```

**Why whole-component availability, stated narrowly.** A donor that observes
two of a panel's three members *can* teach the relationship within that
observed pair, so the earlier claim that it "cannot teach the relationships
inside the panel" was too strong. The argument that survives is narrower:
**`P(M_a = M_b = M_c = 1)` is not in general recoverable from the pairwise
`J`** — pairwise joint observation only bounds it (`A ≤ min J`, with equality
only in special cases). How often a donor sees a *whole* component at once is
therefore genuinely higher-order information, and `W_comp` is only worth
having if it uses it.

**Correction (B5).** The earlier specification scored donors by
`min_{(f,g) ∈ edges(K_k)} J_j[f,g]`, which **derives availability from pairwise
`J`**. That is a re-reading of pairwise information, not higher-order, and it
overstates availability (it is the upper bound above). The definition above
replaces it: `A_j(K)` is measured directly from donor `j`'s mask, which needs
one new donor summary — the frequency table of observation patterns over the
maskable features (at most `2^m` counts; `m ≤ 7` here). The receiver-side
weight `ω_k` stays pairwise: it weights how often the receiver's gaps co-occur
and makes no higher-order claim.

**Where this can and cannot differ from pairwise scoring:** for a two-feature
component `A_j(K) = J_j[f,g]` exactly, so `W_comp` adds higher-order
information **only on components of three or more features**. Here that is
concrete `{f1,f5,f6}`, wine `{f0,f7,f8}`, protein `{f0,f4,f7}` — and **none on
kin8nm**, whose panels are both pairs. On kin8nm `W_comp` is a
component-weighted pairwise score by construction.

**Single-feature components are handled by the formula, not special-cased:**
a component with no edges has no pairs, so the receiver has no joint gap there
to weight. Its `ω_k` is defined as **0** (the empty sum of `H` over no edges),
so it adds 0 to both numerator and denominator — it **contributes nothing**, by
construction (`A_j` of a single feature is just its observation rate, and is
never used). Consequences to report, not hide:

- A receiver whose group skips a single-feature panel (group B on concrete and
  protein) gets **no W_comp credit for that panel at all**; `W_comp` then
  scores donors only on the multi-feature components present in its mask.
- If **every** component is single-feature (no edge anywhere), the sum is
  empty → `W_comp` undefined → the §7.1 declared fallback (sample-size
  weighting), flagged in the report with the receiver and seed.

Arm: `q = W_comp(i←j)`, same `α, β, γ` as the other score arms.

### 15.3 `graph-similarity-s_graph`

Weighted adjacency `A = [C]₊` on the edges above `τ` (0 elsewhere); normalised
Laplacian `L = I − D^{-1/2} A D^{-1/2}` (an isolated node gets a zero row and
column). Eigenvalues lie in [0, 2]. Take the eigenvalues sorted descending —
**all `m` of them (`k = m` = number of maskable features), so the eigenvalue
count is not a new hyperparameter**:

```
s_graph(i,j) = 1 − ‖λ_i − λ_j‖₂ / (2·√m)          in [0, 1], symmetric
```

This compares **graph structure** rather than entry-by-entry agreement, so two
clients that skip different-but-analogous panels can still look similar (where
`s` in §6 would not). Arm: `q = s_graph(i,j)`.

**⚠️ Open decision — raw spectra discard feature identity.** The Laplacian
spectrum is invariant to relabelling the nodes, so two clients that skip
*different* panels of the *same* size produce **identical spectra** and
`s_graph = 1`. In this design that is exactly the group A / group B contrast
whenever the skipped panels have equal sizes (kin8nm: two pairs), so raw
`s_graph` would call complementary clients identical. Before E4, `s_graph`
must either be **dropped** or made **identity-aware** (for example, compare
Laplacians on the shared, labelled node set rather than their spectra). Not
decided; recorded here so it is not implemented as written.

### 15.4 E4 diagnostic — panel recovery

Per dataset: the Adjusted Rand Index between the receiver's graph components
(isolated features as singleton clusters) and the **true panel assignment**
(§3.2), averaged over receivers and seeds, at `τ = 0.2` and at `τ_perm`.

**Predictions, both recorded before any §15 code exists:**

- **Operative (Claude's):** panel recovery should be **high on every dataset,
  kin8nm included**. The graph is built from the *mask*, and the injector
  installs panel structure by panel membership regardless of feature
  correlation (within-panel φ ≈ 0.83 on every dataset, kin8nm too, §3.5).
- **Superseded and wrong (the user's original):** high on wine, **low on
  kin8nm** because its panels group nearly uncorrelated features (median
  `|corr|` 0.024). Kept for the record: it attributes to the graph a property
  (feature relatedness) that the mask does not carry.

**Consequence:** panel-recovery ARI is a **sanity check on graph construction**,
not a discriminating result — it should be high everywhere, and a low value
means a bug or a threshold problem, not a finding. The **discriminating
question for E4 is whether `W_comp` beats `W_H`**, which should still depend
on whether panels group genuinely related features: requiring a donor to cover
a whole component only pays off if the component's members carry information
jointly (plausible on wine and protein, not on kin8nm).

### 15.5 Out of scope, with acceptance criteria

- **A learned graph encoder remains out of scope** (§14). It may be proposed
  only if it **beats both `W_comp` and `s_graph` using the same inputs** (the
  clients' training-mask summaries) **and the same training budget**.
- **The GRAPE sample–feature backbone is a separate, deferred question.** While
  missingness representations are being compared, the predictor stays the
  mask-aware MLP of §1, so any difference between arms comes from the weights,
  not from the model.
