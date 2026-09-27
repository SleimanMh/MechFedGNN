---
name: run-report
description: Produce the standard MechFedGNN run report after any experiment run. Use this whenever an experiment script finishes, when the user asks "what happened in that run", or when results need to be explained. The report always has the same six sections so runs can be compared across weeks.
---

# Run report

Every experiment run produces ONE markdown file, `results/<exp>/<run_id>/REPORT.md`,
with exactly these six sections in this order. Never reorder, never skip a
section, never add sections. If a section has nothing to say, write "n/a" and
the reason.

Keep it factual. Report numbers and state what they mean mechanically. Do not
speculate about why a method won or lost — that discussion happens with the
user afterwards.

## 1. Data

- Dataset name, `n`, `d`, target column, task (regression + binarised AUC).
- Split sizes: train / validation / test per client, and the split seed.
- Number of clients, and how rows were assigned to them.
- Confirm zero native NaNs before injection.

## 2. Missingness injection

- One row per client: mechanism, rate, `driver_overlap` / `class_spread` /
  `direction`, seed, `target_cols`.
- The maskable column set, stated once, and confirmation that it is identical
  across clients.
- Realised numbers per client: realised missing rate, mean lift, mean phi.
- One sentence stating in plain words what the injector did, e.g. "MAR with
  driver_overlap 1.0 censors the same rows across all maskable columns, so gaps
  travel together."

## 3. Computed parameters per client

For each client, a compact block:

- `r_f` — per-feature missingness rates (vector, 3 dp)
- `H` — joint-absence matrix: report the 5 largest off-diagonal entries with
  their feature pairs, plus the matrix mean
- `J` — joint-observation matrix: same format
- `C` — phi matrix: 5 largest and 5 most negative entries, plus the mean, and
  the count of pairs excluded by the constant-feature convention
- `S` histograms: the characteristics used, the bin edges, and each client's
  normalised histogram
- `p_j` — sample-size share

Write the full matrices to `params/<client>.json` in the same folder and link
them; put only the summaries in the report.

## 4. Client grouping

This is what the user most wants to see. For each receiver, and for each score:

- The directed score matrix (receiver x donor) for `W_H`, `W_C`, `s_ij`
  (missingness similarity), `S` (population similarity), and `Q`.
- The resulting normalised donor weights per arm.
- A one-line grouping summary per receiver, e.g. "receiver c2: W_H favours
  c0 (0.41) and c4 (0.33); S favours c1 (0.52); Q splits 0.30/0.28/0.24/0.18."
- Flag any receiver where a score was undefined and which fallback fired.

State explicitly whether the scores AGREE or DISAGREE on the donor ordering,
per receiver. Disagreement is the interesting case and must be visible.

## 5. Aggregation and headroom

- The headroom block: `loss_local`, `loss_pooled`, `headroom`,
  `pooling_harm` (= `loss_pooled − loss_local` where positive), the number of
  receiver-seeds where pooling is worse than local, and the verdict
  (POOLING HARMS / NO HEADROOM / THIN HEADROOM / HEADROOM OK), per receiver.
  POOLING HARMS means pooling is worse than local beyond sampling noise.
- Aggregation settings actually used: alpha, beta, gamma, lambda_pop, mix
  ratio, adaptation budget.
- Confirm the FedAvg limit unit test passed in this run.

Headroom is a reference, not a gate: never drop or suppress a receiver or
dataset because of it. If any receiver reports NO HEADROOM, put a context note
at the TOP of section 6: the pooled reference did not improve on local there,
so differences between arms there are expected to be small or unstable under
this configuration — a subset-weighted arm may still differ. Likewise a
POOLING HARMS note: there the question is whether an arm avoids the harm.

## 6. Results and ablations

One table per metric, arms as rows, mean +- std across seeds, plus the delta
versus local-only:

- RMSE (primary, regression)
- MAE
- AUC-ROC — computed by binarising the target at the training-fold median and
  ranking the regression predictions. State the positive-class rate.

Arms, always in this order:

```
local-only
fedavg
uniform-donor          (matched self-weight, equal donor weights)
marginal-rate          (score = similarity of per-feature rate vectors)
missingness-similarity (score = s_ij over C matrices)
coverage-W_H
coverage-W_C
population-S
combined-Q
```

Then:

- Per-receiver table for the primary metric: mean, worst client, and the
  fraction of clients harmed relative to local-only.
- Harm from averaging, per receiver, at both timepoints: `local-only − fedavg`
  and `local-only − uniform-donor` (negative = the averaging arm is worse).
  Then, for every arm, how many receivers and receiver-seeds it harms relative
  to local-only. (E1's primary question — CLAUDE.md §10.)
- Measured transfer benefit `U[i<-j]` per receiver-donor pair, at timepoint 1
  (immediately after mixing) and timepoint 2 (after the equal adaptation
  budget).
- Score-vs-`U` agreement: for each score, in how many receiver-seed
  combinations did its donor ordering match the ordering by measured `U`.

Close with a short "reading" block using this fixed table, and nothing beyond
it:

| Result | Interpretation |
|---|---|
| A score's ordering tracks `U` | That signal carries information about useful transfer |
| No score tracks `U` | The proxies miss what matters in this setting |
| An arm beats local-only and uniform-donor | Its weighting is doing real work |
| An arm beats local-only but not uniform-donor | The gain is from collaborating at all, not from the weights |
| Gains at timepoint 1 but not timepoint 2 | The benefit may be limited to initialisation |
| NO HEADROOM verdict | The pooled reference did not beat local here; a subset-weighted rule still might — read it beside the arm results |

## Machine-readable companion

Alongside `REPORT.md`, always write:

- `results/<exp>/<run_id>/config.json` — the complete frozen config
- `results/<exp>/<run_id>/metrics.csv` — one row per (seed, receiver, arm,
  metric, timepoint)
- `results/<exp>/<run_id>/scores.csv` — one row per (seed, receiver, donor,
  score_name, value) plus the measured `U`
- `results/<exp>/<run_id>/params/<client>.json` — full r, H, J, C, histograms

`run_id` is `<date>_<git short sha>_<config hash>` so two runs are never
confused.
