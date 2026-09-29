# E5 — population compatibility under structured missingness

**Protocol, written before any E5 prediction result is inspected.** Declared
items below are fixed at commit time and not revised afterwards. E5 is a **new
protocol**, not the E3 described in `CLAUDE.md` §10 (which is still unrun); it
reuses E1's model, preprocessing and analysis plan but a different client
construction. E2, E3 and E4 remain unrun.

## 1. Question

> When clients have different covariate distributions, does population
> similarity `S` improve prediction, and does joint-missingness coverage add
> value beyond population similarity and marginal availability?

E1 and E1M used approximately homogeneous populations in their main
constructions, so their weak result for `S` does **not** establish that `S` is
ineffective under population shift. E5 tests that directly.

**What E5 adds beyond E1/E1M.** E1/E1M held populations homogeneous and varied
missingness; every score arm was equivalent to equal donor weights (240/240
corrected contrasts inside ±2 %). E5 varies population *and* missingness
factorially, adds a **marginal-only coverage** arm so pairwise information can
be isolated from marginal availability, and gives `S` a population difference
that actually exists. It does not revisit E1/E1M's conclusions, which stand for
the configurations they tested.

## 2. Factorial design

| Condition | Population | Missingness |
|---|---|---|
| P0M0 | approximately random client assignment | independent cell masking |
| P0M1 | approximately random client assignment | structured panel masking |
| P1M0 | covariate-based, overlapping | independent cell masking |
| P1M1 | covariate-based, overlapping | structured panel masking |

Records, client assignment and train/val/test splits are **identical across M0
and M1** for a given (dataset, P, seed): masks are drawn after assignment and
do not feed back into it. P0 and P1 necessarily differ in assignment — that is
the manipulation.

**No coupled MAR masking in E5.** Both M0 and M1 are value-independent, so
population shift and value-dependent removal are not introduced together.

## 3. Missingness construction (value-independent, matched counts)

**M1 — structured panel masking.** Panels are E1's (frozen, from the design
rows). Each client's group profile sets the per-panel ordering probability:
group A rarely orders the first half of the panel list (`p_rare = 0.2`), group
B the second half; the rest at `p_usual = 0.9`; `jitter = 0.05` so within-panel
φ stays below 1 and incomplete rows do not always lose a whole panel. Counts
are **exact per fold**: for each panel, exactly `round(p·n)` rows ordered, and
exactly `round(jitter·n_ordered)` of those lost per member feature.

**M0 — independent cell masking, by column permutation of M1.** For each
client and fold, each column of the M1 mask is permuted independently. This is
the "independently randomized column arrangement" construction: it preserves
**per-feature missing counts exactly** (identical to M1, by construction, not
merely in expectation) and destroys row-level association.

*Designed consequence, to be verified in the construction checks:* per-feature
rates `r`, and therefore the marginal-only coverage score `W_marg`, are
**identical between M0 and M1** for the same client and seed; `S` is identical
too (it reads never-masked features). Only the joint quantities `H`, `J`, `C` —
and hence `W_H` and `Q_H` — change. M0 is therefore an exact control for
pairwise structure.

*Finite-sample constraint.* Exact counts move in steps of one row, so realised
rates across clients can differ by about `1/n_fold`. The declared tolerance
below is applied unchanged; `CLAUDE.md` §16.1 records the earlier VOID cells
caused by this and the candidate rule that was **not** applied.

## 4. Population construction (target-free)

**Partition characteristic — declared rule.** Among the `always_observed`
features (never masked, hence reliably observed), on the **design rows only**:
the feature with the most distinct values; ties broken by lowest column index.
This is **independent of the target**. The existing `partition_col` in
`data/clients.py` picks the always-observed feature most correlated with the
target and is **not used in E5**.

**P1 assignment.** Split the client pool at the partition characteristic's
median (computed on pool rows, design excluded). Client `k` draws a
`frac_low[k]` share of its rows from the LOW half and the rest from HIGH, with
**`frac_low = [0.80, 0.20, 0.80, 0.20]`** — the declared assignment strength.
Every client therefore holds both halves: distributions overlap, none is
disjoint, and the task is not made extreme.

**P0 assignment.** E1's corrected path: random permutation of the pool into
four near-equal parts.

**Both P conditions:** duplicate groups are assigned whole (a group's location
follows its representative row), and the design / train / val / test separation
of §16 is preserved. Assignment is performed at group level so no group is
split.

**Cross-cutting, declared.** Missingness groups are clients {0,1} vs {2,3};
`frac_low` makes population-LOW clients {0,2} and population-HIGH {1,3}. The
two groupings cross-cut, so population similarity and missingness structure
point at different donors.

*Scope.* This creates covariate heterogeneity from an existing tabular dataset
by re-assigning its own records. It does **not** reproduce real hospital
populations and does **not** guarantee identical conditional outcome
distributions across clients — the marginal covariate distributions differ by
construction, and `P(y|x)` is only unchanged to the extent the dataset's own
relationship is stable across the split.

## 5. `S` configurations (declared)

Shared histogram bins are the frozen design-split deciles already in
`configs/roles*/`. Client histograms are estimated from **training rows only**.

- **Primary controlled configuration:** characteristics = all
  `always_observed` features, which **includes** the partition characteristic.
  Establishes whether `S` can work when it directly measures the constructed
  shift.
- **Robustness configuration:** characteristics = all `always_observed`
  features **except** the partition characteristic. Tests whether any effect
  depends on that privileged information.

Neither bins nor characteristics are chosen by test performance.

## 6. Arms (8) and scores

| Arm | Score `q` |
|---|---|
| `local-only` | — |
| `fedavg` | α = 0, γ = p_i |
| `uniform-donor` | equal donor weights, γ = 0.5 |
| `coverage-marginal` | `W_marg(i←j) = Σ_f r_i[f]·(1−r_j[f]) / Σ_f r_i[f]` |
| `coverage-W_H` | `W_H(i←j)` (pairwise) |
| `population-S` | `S(i,j)` |
| `combined-Q` | `Q_H = 0.5·W_H + 0.5·S` |
| `combined-Q-marginal` | `Q_marg = 0.5·W_marg + 0.5·S` |

`W_marg` is new and is the matched marginal analogue of `W_H`: same functional
form, marginal availability instead of joint. `combined-Q-marginal` is the
essential control — if `Q_H` improves, it says whether pairwise information
contributes beyond marginal availability plus population similarity.

`λ_pop = 0.5` fixed; **no tuning sweep**. Self-weight (γ = 0.5), local budget
and adaptation budget (**200 and 10 steps**, chosen in Stage B by the declared
method-agnostic rule below) and donor participation are identical across all
score arms and `uniform-donor`. `marginal-rate`,
`missingness-similarity` and `coverage-W_C` from E1 are **not** run in E5: they
do not serve these comparisons and would add multiplicity.

No GNNs, attention, learned gates or reconstruction loss.

## 7. Primary comparisons

Reported separately by dataset and condition, as paired seed-level contrasts:

1. `Q_H` vs `W_H` — does population similarity add value?
2. `Q_H` vs `S` — does coverage add value?
3. `Q_H` vs `Q_marg` — **does pairwise information add value beyond marginal availability + S?**
4. `S` vs `uniform-donor` — does the population signal help?
5. `Q_H` vs `uniform-donor` — does the combined method help overall?

Each is also compared **across conditions** (P0 vs P1, M0 vs M1). Effect
estimates and uncertainty are reported, not only decision labels.

## 8. Analysis (unchanged from `CLAUDE.md` §8.1)

Δ% = 100·(RMSE_a − RMSE_b)/RMSE_b in original units, macro-averaged over
receivers within a seed; mean and 95 % t-interval over seeds; **seed is the
replication unit**. Practical margin **δ = 2 %**, retained for comparability —
not re-tuned. Also reported: immediate (t1) and post-adaptation (t2) results,
per-client changes and harmful transfer, score variation and realised weight
variation, donor-score agreement with measured transfer benefit `U`, and the
validation-selected donor diagnostic. Test-measured donor utilities stay
**retrospective**: never used to build scores or choose settings.

## 9. Stages and seeds (declared)

- **Stage A — synthetic sanity check.** One shared outcome function across
  clients, overlapping but different covariate distributions, value-independent
  missingness, and a relationship whose predictive relevance varies across the
  covariate space. Checks the pipeline and that the intended population
  difference exists. Success for `S` or `Q` is **not** required.
- **Stage B — development run.** Concrete and Wine, **development seeds
  `[2, 3, 5]`**, training/validation only. Numerical stability, training
  progress, runtime, and a learning curve to pick the training budget. The
  budget is **not** chosen to maximise separation between methods.

  *Rule, declared before the curves were read:* `local_steps` = the smallest
  grid point whose mean validation RMSE of **local-only** is within 1 % of that
  curve's minimum; `adapt_budget` = the same rule applied to the
  **uniform-donor control**. Each dataset's curve is normalised by its own best
  before averaging. Neither curve involves a score arm, so the budget cannot be
  chosen to favour one. **Result: `local_steps = 200`, `adapt_budget = 10`**
  (both interior to their grids; `adapt_budget` coincides with E1's
  independently chosen value).

  *Two corrections made during Stage B, both before any test fold was touched:*
  the grids were extended once because the first pass selected their
  boundaries; and an aggregation bug was fixed — the local curve had been
  normalised by the maximum **across datasets at each step** instead of by each
  dataset's own minimum **across steps**, which made the average track the
  wine/concrete scale ratio and spuriously select the smallest budget.
- **Stage C — frozen evaluation.** Concrete, Wine, kin8nm, Protein.
  **Evaluation seeds `[211, 223, 227, 229, 233, 239, 241, 251, 257, 263]`** —
  disjoint from E1/E1M's seeds and from the development seeds. Protocol and
  seed list frozen before test outcomes are inspected.

Fresh seeds do **not** make these datasets an untouched benchmark: they were
studied in E1/E1M and that history stays visible.

## 10. Construction checks (before training)

Per dataset, client, fold and condition: sample size and duplicate-group
separation; per-feature missingness rates; joint-absence `H`; within- and
between-panel `C`; counts of rows with zero / one / multiple missing features;
partial-panel missingness; population histogram differences; realised `W_marg`,
`W_H`, `S`, `Q_H`, `Q_marg`; resulting aggregation weights and any fallbacks.

**Matched-count invariant (corrected before any E5 run; see note).** For every
dataset, client and fold: the per-feature missing **counts are identical
between M0 and M1**, exactly, not merely within a tolerance. Verified by direct
comparison of the two masks' column counts. Equivalent consequences, also
verified: `r`, `W_marg` and `S` are identical between M0 and M1.

Cross-client rate differences are **not** bounded in E5 — they are the design:
clients carry different group profiles (`p_rare` vs `p_usual` per panel), which
is what gives any coverage score something to detect. They are reported
descriptively.

> **Note on a corrected check.** An earlier draft of this section required
> `max |r_i,f − r_j,f| ≤ 0.01` *across clients* within a fold. That is E1M's
> invariant, where clients have matched marginals by construction; importing it
> into E5 was an error, and it would have forbidden the very heterogeneity E5
> needs. It was corrected after the first construction check and **before any
> E5 prediction result was produced or inspected**. The E1M tolerance itself is
> unchanged, and the E1M VOID cells recorded in `CLAUDE.md` §16.1 stand.

**If a construction fails, it is reported and the construction is fixed before
any prediction result is examined.** No dataset is dropped because its
prediction results are unfavourable.

*Known property, reported not tuned.* For rarely-ordered panels (`p_rare` =
0.2) the jitter mechanism acts on the few ordered rows, so partial-panel rows
are a small share of incomplete rows (~3 %). Within-panel `C` is ≈ 0.95 for
those panels and ≈ 0.65 for usually-ordered ones — strong association, not the
degenerate `C` = 1. `jitter` stays at E1's 0.05; it is not adjusted to make
this number look better.

**A construction passes because it satisfies the design — not because a
proposed score prefers a particular donor.**

## 11. Interpretation, declared in advance

- **`S` helps but pairwise coverage adds nothing** (`Q_H` ≈ `Q_marg`, both >
  `uniform`): evidence for population compatibility, not co-missingness.
- **Pairwise adds beyond the marginal combination** (`Q_H` < `Q_marg` beyond
  δ): evidence worth confirming in another setting.
- **No method improves meaningfully:** the negative finding is retained, and we
  identify whether population shift, donor opportunity or score–utility
  alignment was weak.
- **Effects appear only in the primary `S` configuration** (not the robustness
  one): reported clearly as dependence on privileged information.

---

## 12. Outcome (recorded 2026-09-29; `results/e5/analysis.txt`, 32 runs)

Construction checks passed on all 16 dataset × condition cells (0 count
mismatches, 0 duplicate groups split). Stage A and Stage B passed; budget
frozen at `local_steps = 200`, `adapt_budget = 10`.

**All 320 primary contrasts are negligible** — 160 at t1 and 160 at t2, every
95 % interval inside ±2 %, and in fact inside ±0.7 %. This holds in both `S`
configurations and in all four conditions.

By comparison (all t2, primary `S`):

| Comparison | Result |
|---|---|
| `Q_H` vs `W_H` — does `S` add value? | negligible everywhere (−0.23 … +0.14 %) |
| `Q_H` vs `S` — does coverage add value? | negligible everywhere (−0.06 … +0.14 %) |
| `Q_H` vs `Q_marg` — **does pairwise add beyond marginal + `S`?** | negligible everywhere (−0.15 … +0.02 %) |
| `S` vs `uniform-donor` — does the population signal help? | negligible in all 32 cells (−0.03 … +0.05 %) |
| `Q_H` vs `uniform-donor` — does the combined method help? | negligible everywhere (−0.07 … +0.16 %) |

**This is the "no method improves meaningfully" branch of §11.** Per that
branch, we identify which precondition was weak:

- **Population shift: NOT weak.** `S`'s score range across donors rises from
  0.004–0.031 (P0) to 0.113–0.363 (P1), and its realised weight range from
  0.001–0.012 to 0.047–0.164. The manipulation did what it was built to do.
  In the **robustness** configuration that variation collapses (concrete
  0.149 → 0.052), confirming the partition characteristic was carrying it —
  yet the prediction result is unchanged in both configurations, so **no
  effect depends on the privileged information** (there is no effect to
  depend on it).
- **Donor opportunity: NOT weak on concrete and wine.** `uniform-donor` beats
  `local-only` by 3.4–5.3 % on wine (meaningful in all four conditions) and
  1.5–3.5 % on concrete (unresolved). It is weakly harmful on kin8nm and
  protein (+0.25 … +0.49 %, harming 25–38 of 40 receiver-seeds).
- **Score–utility alignment: WEAK.** Top-donor agreement with measured `U` is
  at or below the 1/3 chance rate (≈ 13/40) in most cells, with Kendall τ near
  zero or negative — notably kin8nm (τ −0.18 … −0.49 for every score).
  Exceptions are mild: protein `W_H` 16–22/40 (τ ≈ 0) and wine P1 17–20/40
  (τ ≈ +0.2).
- **Weight dispersion is not the bottleneck here.** Unlike E1, the scores do
  produce non-uniform weights (`W_marg` range ≈ 0.22, `W_H` ≈ 0.23–0.36 of
  the weight simplex). They simply do not point at donors that help.

**Selector diagnostic.** The validation-selected single donor beats
`local-only` meaningfully on wine (−2.9 … −5.4 %) but **never** beats
`uniform-donor` (negligible or unresolved in all 16 cells) — the same pattern
E1 found.

**Answer to the E5 question.** Under constructed, overlapping population shift
with value-independent missingness: **population similarity did not improve
prediction**, and **joint-missingness coverage added nothing beyond marginal
availability plus population similarity**. The earlier weak result for `S` in
E1/E1M is therefore not explained by those experiments' homogeneous
populations — `S` remains ineffective when a real population difference is
present and `S` can see it directly.

**Scope.** This is a negative result for these scores, this predictor, this
aggregation rule and these four tabular datasets at K = 4. It does not rule
out population compatibility as a concept, other predictors, other
aggregation rules, or learned collaboration methods.
