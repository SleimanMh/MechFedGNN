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
| Data | GRAPE-9 UCI suite, fetched by a script you write (§2). Start with **concrete, housing, wine, naval**. Complete data + injected missingness. |
| Task | Regression (MSE). AUC-ROC additionally reported by binarising the target at the **training-fold median** and ranking the regression predictions. No second model. |
| Backbone | Mask-aware MLP: input `concat([x_zero_filled, M])`, hidden (64, 32), ReLU, 1 output. |
| Loss | **`L = L_pred` only.** No reconstruction term, no reconstruction head. |
| Clients | `K = 6` in `G = 2` latent groups (§4). **Every client acts as receiver in turn** — gives per-client and worst-client results for free. |
| Splits | train / val / test = 60/20/20 per client, split first. Summaries use **training rows only**. Tuning on val. Test touched once. |
| Standardisation | Receiver's own training-fold statistics. Never pooled, never from test. |
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
- CLI: `--out ./raw`, `--datasets concrete housing wine naval`

## 3. `data/inject.py` — panel-based missingness

This is the core of the setup and the part that must look **realistic**. Random
per-cell missingness does not resemble real records. In practice features go
missing in **panels**: a lab panel is either ordered for a patient or it is not,
and when it is not, every feature in that panel disappears together. That is
where genuine co-missingness comes from, and it is exactly what the C matrix is
meant to detect.

### 3.0 Design split (`data/design.py`)

A fixed 10% of rows (seed 0) is **excluded from every client** and used only
for column roles (§3.1), panel assignment (§3.2) and histogram bin edges
(§4.3). Frozen design choices therefore never see a row any client trains,
tunes or tests on. Tested: no design row ever appears in a client.

### 3.1 Column roles (computed once per dataset, frozen in config)

- Compute `|corr(f, target)|` on the design-split rows.
- Features constant on the design rows (e.g. naval f8, f11) are listed as
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
panel (sample failure, transcription error). It keeps within-panel φ below 1,
which is what real data looks like; φ = 1.0 exactly is a tell-tale of synthetic
data.

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
- `driver_overlap ∈ [0,1]` (MAR only) — the fraction of panels that share one
  driver score `z_r`. At 0 each panel has its own independent driver, so panels
  are skipped independently; at 1 all panels are skipped on the same rows, so
  co-missingness spans panels as well as within them. **This is the
  dose-response knob.**
- `class_spread ∈ [0,1]` (CD-MNAR only) — how unequal the per-outcome-bin rates
  are. At 0 it collapses to MCAR.

Both knobs at 0 must return the setup to independent, marginal-equivalent
missingness. That collapse is the null condition — verify it.

### 3.5 Diagnostics the injector must report (`data/validate_injector.py`)

For each configuration, over several seeds:

- realised overall missing rate, and per-feature rates
- **within-panel mean φ** — should be high (≈ 0.7–0.95 with jitter 0.05)
- **cross-panel mean φ** — should be ≈ 0 under MCAR and rise with
  `driver_overlap` / `class_spread`
- lift matrix mean and max
- a **realism check** table: fraction of rows with 0, 1, 2, … missing panels.
  If almost every row is either complete or missing everything, the setup is
  unrealistic — flag it.

Two hard checks, printed with a PASS/FAIL verdict:

- **Null:** under MCAR with `driver_overlap = 0`, cross-panel φ must be within
  sampling noise of 0. Use the analytic SE `sqrt((1-p²)/(n·p²))` rather than a
  fixed tolerance — a fixed tolerance just measures `n`.
- **Dose-response:** sweeping `driver_overlap` 0 → 1 must raise cross-panel φ
  monotonically. Same for `class_spread`.

Save the output to `results/injector_validation/<dataset>.txt`.

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

`K = 6` clients: 3 per group. This is the **ground truth**. Two clients in the
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

Clients get equal shares by default. One config knob, `receiver_fraction`,
shrinks whichever client is currently the receiver to a fraction of its normal
size (default 0.3). A receiver with abundant data has nothing to learn from
anyone, which produces an uninterpretable null — see the headroom check (§8).

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

**Headroom, run before every comparison.** On the receiver's test fold, under
its own mask: `loss_local` (own training rows only) vs `loss_pooled` (oracle
trained on pooled training rows of all clients, evaluated under the receiver's
mask — the upper bound for parameter aggregation). `headroom = loss_local −
loss_pooled`. Verdicts: `headroom/loss_local ≤ 0.02` → **NO HEADROOM**;
`≤ 0.10` → THIN; else OK. Print above every results table. Pooling is a
diagnostic only — never an arm, never a weight source.

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

Each runs on concrete, housing, wine and naval, **reported per dataset, never pooled**,
every client as receiver in turn, 10 seeds. Each produces a report via the
`run-report` skill.

**E1 — signal comparison (the main one).**
`population_homogeneous`, MAR with `driver_overlap = 0.5`, rate 0.3, the latent
group structure of §4.1. All nine arms.
Questions: does any score's donor ordering track measured `U`? Does any arm beat
both `local-only` and `uniform-donor`?
**Also report group recovery:** cluster clients into 2 groups by each symmetric
score (`rate`, `s`, `S`) and report the Adjusted Rand Index against the true
group labels. `s` should recover the missingness groups; `rate` should do worse
(clients differ in structure, not in overall rate); `S` should be at chance under
`population_homogeneous`.

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
missing-panels-per-row realism table.

**Stage 1 — quantities and scores.** Write the §13 tests first (they fail), then
`signatures.py`, `scores.py`, `kernel.py` until green.
**STOP** — show full pytest output.

**Stage 2 — clients, model, loop, headroom, report.** Then pilot the adaptation
budget: scan `{0, 10, 25, 50, 100}` on one seed and the validation fold, pick the
smallest that separates arms stably, freeze it in `defaults.yaml` with a printed
justification. Also sweep `receiver_fraction ∈ {0.2, 0.3, 0.5}` and pick the
largest that still gives HEADROOM OK.
**STOP** — show headroom verdicts per client and the chosen budget and fraction.

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
- headroom detects both states: OK at the chosen `receiver_fraction`,
  NO HEADROOM when the receiver gets a full-sized sample
- determinism: same seed → identical CSVs

## 14. Do not build

Any GNN or GRAPE backbone · PyTorch Geometric · reconstruction loss or head ·
attention · learned encoders · sharpening (β ≠ 1) · Ditto, FedProx, Clustered
FL · DARN, Cafe, FedSaC · eICU or MIMIC · multi-round federation · differential
privacy · wandb · any abstraction layer "for later extensibility".

A negative result reported clearly is a valid outcome of this round. Do not tune
toward a win.
