# E5 selection diagnostic — EXPLORATORY

*Post hoc analysis of existing E5 artefacts. No retraining, no injector change,
no tuning, no new model evaluation. All E5 runs and results are preserved
unchanged. Outputs: `results/e5/diagnostic.txt`, `results/e5/diagnostic_pop.txt`.*

> **Question.** Is there a useful donor-selection opportunity beyond uniform
> aggregation, and can the available validation data reliably identify it?

Unit of replication is the **seed** (n = 10); receivers are averaged within a
seed before any interval is formed. Receivers, donor pairs and conditions are
not treated as independent replications. Margin δ = 2 %.

**The test-best donor is an optimistic retrospective reference** for the
available single-donor candidates — not a deployable method, and not a ceiling
for aggregation methods in general.

## 1. Collaboration benefit vs selection benefit

All figures are Δ% versus **uniform-donor**, post-adaptation (t2), primary `S`
configuration, with self-weight (γ = 0.5) and adaptation budget (10 steps)
matched across every quantity. Negative = better than uniform.

| dataset | local-only | random donor | test-best donor (retro) | val-selected donor | val-selected donor *or* local |
|---|---|---|---|---|---|
| concrete P0M0 | +3.80 | +0.86 | **−2.51** | −0.42 | +0.35 |
| concrete P0M1 | +1.84 | +0.44 | **−3.16** meaningful | −0.39 | −0.10 |
| concrete P1M0 | +3.25 | +0.89 | **−2.09** | +0.47 | +0.85 |
| concrete P1M1 | +2.32 | +0.34 | **−2.90** | −1.18 | −0.42 |
| wine P0M0 | +4.87 meaningful | +0.49 | −1.54 | +0.73 | +0.73 |
| wine P0M1 | +5.92 meaningful | +0.35 | −1.66 | −0.17 | +0.10 |
| wine P1M0 | +4.52 meaningful | +0.36 | −1.59 | +0.30 | +0.36 |
| wine P1M1 | +3.73 meaningful | +0.14 | −1.52 | +0.51 | +0.80 |
| kin8nm (4 cond.) | −0.36 … −0.48 | +0.01 … +0.06 | −0.54 … −0.67 | −0.34 … −0.55 | −0.50 … −0.62 |
| protein (4 cond.) | −0.25 … −0.47 | +0.00 … +0.05 | −0.18 … −0.32 | −0.16 … −0.30 | −0.33 … −0.49 |

Everything except the marked cells is **negligible** (interval inside ±2 %);
the concrete cells against local-only and the test-best donor are largely
**unresolved**, i.e. *no meaningful difference was demonstrated* — not
"no difference exists".

Three things follow.

1. **Uniform aggregation is barely better than a random donor** (+0.00 … +0.89 %
   everywhere). Whatever uniform aggregation achieves, it is not achieved by
   being a clever combination.
2. **A real single-donor opportunity exists only on concrete** (test-best
   −2.1 … −3.2 %). On wine it is ≈ −1.5 %, on kin8nm ≈ −0.6 %, on protein
   ≈ −0.2 %.
3. **The two selectors behave differently, as expected, and should not be
   conflated.** "Which donor" and "whether to collaborate" are different
   decisions: on kin8nm and protein, adding `local-only` to the candidate set
   improves the selector (−0.50 … −0.62 vs −0.34 … −0.55 on kin8nm), because
   there declining to collaborate is often right. On concrete it *hurts*
   (+0.35 vs −0.42 in P0M0), because the selector then sometimes declines a
   collaboration that would have helped.

At t1 (immediate, pre-adaptation) the picture is noisier and the random-donor
penalty is larger (+1.7 … +2.9 % on concrete and wine), i.e. adaptation
compresses differences between donors — consistent with E1.

## 2. Stability of donor preferences (t2, validation vs test)

Chance agreement for "validation-best donor is also test-best" is **1/3** with
three donors. No exact ties occurred in any cell (0 of 160 receiver-seeds per
condition).

| dataset | τ(val, test) | val-best = test-best | val gap 1st→2nd | disagree on "improves over local" |
|---|---|---|---|---|
| concrete | +0.15 … +0.38 | 42 – 57 % | 3.2 – 3.5 % | 37 – 43 % |
| wine | −0.08 … +0.08 | **30 – 38 % (chance)** | 2.0 – 2.2 % | 23 – 31 % |
| kin8nm | +0.30 … +0.55 | 57 – 72 % | 0.6 – 0.9 % | 21 – 39 % |
| protein | +0.55 … +0.80 | 72 – 88 % | 0.2 – 0.4 % | 8 – 13 % |

**The relationship is inverted from what a working selector needs.** Where the
opportunity is largest (concrete, ≈ 3 %), validation identifies the right donor
only about half the time — despite the validation winner being separated from
the runner-up by 3.3 %, i.e. validation is *confidently* wrong. Where
validation is reliable (protein, 72–88 % agreement, τ ≈ 0.7), the opportunity
is ≈ 0.2 % and not worth acting on. Wine is the worst case: donor selection is
**at chance**, and the validation gap (≈ 2 %) is pure noise.

Validation and test also disagree on the simpler binary question — does this
donor beat local-only? — in 8 – 43 % of receiver-seeds.

### Donor rankings change after adaptation

Test-fold donor ranking, before (t1) vs after (t2) the 10-step adaptation:

| dataset | τ(t1, t2) | same best donor |
|---|---|---|
| concrete | +0.38 … +0.50 | 52 – 68 % |
| wine | +0.33 … +0.47 | 55 – 60 % |
| kin8nm | +0.55 … +0.65 | 70 – 85 % |
| protein | +0.33 … +0.70 | 48 – 82 % |

The best donor changes across the adaptation step in roughly a third to a half
of receiver-seeds. **Any donor-utility model must therefore target the
adaptation protocol actually used**; a utility estimated pre-adaptation is
estimating a different quantity.

## 3. What the population construction establishes

Three claims must be kept apart.

**(a) A difference in covariate distributions — established.** Under P1 the
below-median share of the partition characteristic moves to ≈ 0.80/0.20/0.79/0.19
across clients with support overlap preserved, and `S`'s realised weight range
rises from 0.001–0.012 (P0) to 0.047–0.164 (P1).

**(b) A difference relevant to predicting the target — partially assessable,
and dataset-dependent.** From development artefacts only (`|corr(f, target)|`
on the design rows, `configs/roles_corrected/corr/*.yaml`):

| dataset | partition characteristic | \|corr(target)\| | rank among all features |
|---|---|---|---|
| concrete | f0 | **0.592** | **1 of 8** |
| wine | f7 | 0.137 | 5 of 11 |
| kin8nm | f0 | 0.158 | 3 of 8 |
| protein | f3 | 0.161 | 3 of 9 |

So on concrete the split is on the single most target-correlated feature —
plausibly task-relevant. On the other three it is a middling feature, and a
linear marginal correlation is weak evidence of task relevance in any case.
**A large histogram difference does not establish task relevance**, and the
saved artefacts cannot settle it beyond this correlation.

**(c) A difference that predicts donor usefulness — NOT established.**
Comparing measured transfer benefit `U` for donors in the receiver's own
population half versus the other half (P1 only, % of local-only RMSE):

| dataset | P1M0 | P1M1 |
|---|---|---|
| concrete | −0.31 [−1.33, +0.70] | +0.78 [−0.54, +2.10] unresolved |
| wine | +0.85 [−0.19, +1.88] | +0.57 [−0.31, +1.45] |
| kin8nm | −0.35 [−0.53, −0.18] | −0.41 [−0.74, −0.09] |
| protein | +0.06 [−0.01, +0.12] | +0.10 [+0.01, +0.19] |

All negligible except one unresolved cell, and the sign is **negative** on
kin8nm (same-population donors transfer slightly *worse*). Identical in both
`S` configurations.

### Audit of the "target-free" description — the description was inaccurate

The partition characteristic was chosen by a target-free rule (most distinct
values, ties by column index) **from the `always_observed` set**. But
`always_observed` is defined as the complement of `maskable`, and `maskable`
in the `corr` mode used by E5 is *the half of features with the lowest
|corr| with the target*. **The eligible set is therefore the high-|corr|
half — it was selected using target correlations.**

The accurate description is: *a target-free choice conditional on a
target-informed eligible set.* On concrete this matters most — the rule landed
on the most target-correlated feature in the dataset.

This is a **wording correction, not a result correction**. E5 is not re-run
for it. A genuinely target-free variant already exists (`maskable_mode =
"random"`, `configs/roles_corrected/random/`) and would be the right basis if
this construction is used again.

## 4. Audit of counts and interpretation

| Check | Result |
|---|---|
| 320 primary contrasts include both `S` configurations and both timepoints | **Verified.** 5 comparisons × 4 datasets × 4 conditions × 2 `S` configs × 2 timepoints = 320; label counts 160 negligible at t1 + 160 at t2 |
| 16 dataset–condition cells per `S` configuration | **Verified.** 32 runs, 32 distinct (dataset, condition, `S` config), 16 per configuration |
| "No meaningful improvement demonstrated" used where unresolved | **Verified** in `PROTOCOL_E5.md` §12 (concrete labelled unresolved); this document uses the same wording |
| Marginal coverage distinguished from marginal-rate similarity | **Clarified below** |
| Column permutation removes association *in expectation* | **Corrected below** |

**Marginal coverage vs marginal-rate similarity.** These are different
quantities and only the first is in E5. `W_marg(i←j) = Σ_f r_i[f]·(1−r_j[f]) /
Σ_f r_i[f]` is a **directed coverage** score — "does the donor observe what the
receiver lacks". E1's `marginal-rate` was `0.5·(1 + cos(r_i, r_j))`, a
**symmetric similarity** of rate vectors. E5 ran `W_marg` and did **not** run
`marginal-rate`; no E5 claim rests on the E1 baseline.

**Three corrections to `PROTOCOL_E5.md` §12,** applied and marked there:

1. "every interval … inside ±0.7 %" → the true maximum bound over the 320
   contrasts is **±0.81 %** (interval [−0.56, +0.81]).
2. "target-free partition characteristic" → *target-free selection conditional
   on a target-informed eligible set* (§3 above).
3. M0 "destroys row-level association" → independent column permutation
   preserves per-feature counts **exactly** but removes imposed cross-column
   association only **in expectation**. Measured residual in M0 (seed 211,
   training folds): within-panel `C` −0.03 … +0.10, between-panel −0.08 …
   +0.11, against +0.63 … +0.95 within-panel in M1. The contrast is large, but
   M0 is not an exactly-zero-association control at finite n.

## 5. Decision-oriented reading

Findings coexist; each dataset is assigned every row it supports.

| dataset | Situation supported | Evidence |
|---|---|---|
| **concrete** (all 4 conditions) | **Best donor improves, but validation cannot identify it reliably** → investigate evaluation noise / selection reliability | test-best −2.1 … −3.2 % vs uniform; val-best = test-best only 42–57 %; validation gap 3.2–3.5 % yet wrong ~half the time |
| | Donor rankings change substantially after adaptation | same best donor across adaptation 52–68 % |
| | Population differences visible; task relevance plausible but unproven; does not predict donor usefulness | partition characteristic rank 1 by \|corr\| (0.592); `U` same- vs other-population negligible/unresolved |
| **wine** | **Limited opportunity** for this single-donor intervention *and* validation cannot identify it | test-best only −1.5 … −1.7 %; val-best = test-best **30–38 %, i.e. chance**; val-selected is *worse* than uniform (+0.3 … +0.7) |
| | Collaboration itself clearly helps | local-only +3.7 … +5.9 % worse than uniform, meaningful in all 4 conditions |
| | Donor rankings change after adaptation | same best donor 55–60 % |
| | Population differences visible but task relevance unknown | partition characteristic rank 5 of 11 (\|corr\| 0.137) |
| **kin8nm** | **Limited opportunity**; selection is reliable but there is little to win | test-best −0.54 … −0.67 %; val-selected recovers most of it (−0.34 … −0.55); agreement 57–72 % |
| | Collaboration does not help here | local-only is −0.36 … −0.48 % *better* than uniform (negligible) |
| | Population differences visible but task relevance unknown; same-population donors transfer slightly worse | `U` same − other −0.35 … −0.41 |
| **protein** | **Limited opportunity**; selection is reliable (best case) but the prize is ≈ 0.2 % | test-best −0.18 … −0.32 %; val-selected −0.16 … −0.30; agreement 72–88 %, τ 0.55–0.80 |
| | Collaboration does not help here | local-only −0.25 … −0.47 % better than uniform |
| | Population differences visible but task relevance unknown | partition characteristic rank 3 of 9 |

**No cell supports "validation selects useful donors, but existing scores do
not."** That row would justify task-dependent donor compatibility as a
candidate; the evidence does not reach it. On concrete, where the opportunity
is real, validation *cannot* find it; where validation works, there is nothing
worth finding.

## 6. Recommended next experiment (one)

**Donor-utility estimability study — "can donor usefulness be estimated at all,
under the deployed adaptation protocol?"**

*The failure it addresses.* Every experiment so far (E1, E1M, E5) asked whether
a **hand-built score** ranks donors well, and answered no. This diagnostic
shows a prior question was never settled: on concrete a 2–3 % selection
opportunity exists, yet a held-out validation fold — the most direct possible
estimator of donor utility, using the true loss rather than any proxy —
identifies the right donor only ~50 % of the time while appearing confident
(3.3 % gap). If direct measurement cannot estimate donor utility reliably, **no
score built from mask summaries can**, and the E1/E1M/E5 negatives are
explained by estimator noise rather than by co-missingness being uninformative.
That distinction changes what to build next, so it should be settled first.

*Design.* Existing setup, existing datasets, no new model class. For concrete
and wine (where opportunity and collaboration exist) under P0M1 and P1M1:

1. Estimate each donor's utility by **repeated / cross-fitted splits** of the
   receiver's own train+validation data — several disjoint validation folds,
   each scored under the **same adaptation budget used at deployment** (the
   t1/t2 gap above shows the protocol must match), then averaged.
2. Compare selection quality against the current single-split selector, as a
   function of the number of folds and validation size.
3. Report how much of the retrospective test-best gap each estimator recovers.

*Simplest baseline that must be beaten.* **Uniform-donor aggregation** — and,
because uniform ≈ random donor here, a randomly chosen donor is the honest
floor. `local-only` is included in the candidate set for the "collaborate at
all" decision, kept separate from "which donor".

*How joint-missingness information would then be tested.* Once a utility
estimate with known reliability exists, regress measured `U` on two nested
feature sets and compare **held-out prediction of `U`** plus the downstream
loss of selecting by each:

- **Marginal / task-only:** per-feature rates, client sizes, population
  similarity `S`, and task-side summaries.
- **Marginal + joint:** the same plus `H`, `J`, `C`, `W_H`.

Joint information earns its place only if the second set predicts `U` better
out of sample *and* selects better downstream. This keeps co-missingness
central without assuming it helps, and it satisfies the constraint that **any
learned method using joint information must be compared against a learned
method using only marginal or task information** — both arms are learned, on
the same inputs and budget.

*What would make this stop.* If cross-fitted estimation still identifies the
best donor at near chance on concrete, then donor utility is not estimable at
this data scale, and the right conclusion is about evaluation capacity, not
about co-missingness.
