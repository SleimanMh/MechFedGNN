# Adding a signature, score or learner

Each addition touches **one** file plus a registration line. Score calculation,
weight normalization and parameter aggregation are separate operations, so a
new score never requires changing weighting or aggregation.

## Add a scoring strategy

`mechfedgnn/scoring.py`:

```python
def _my_score(receiver, donor):
    """receiver/donor are summaries: {"sig": {"r","H","J","C"}, "hist": {...}, "n_train": int}.
    Return a float, or None if undefined for this pair."""
    num = (receiver["sig"]["H"] * donor["sig"]["J"]).sum()
    den = receiver["sig"]["H"].sum()
    return None if den <= 0 else float(num / den)

SCORES.register("my-score", lambda **kw: _Fn("my-score", _my_score))
```

Then use it: `--arms my-score`, or `cfg["arms"] = [..., "my-score"]`.

Nothing else changes. The declared fallback already handles your `None`: if the
score is undefined for **any** donor, or all donors tie within `NONDISCRIM_TOL`,
the arm falls back to sample-size weighting and records the reason.

**Conventions your score must respect** — `M = 1` observed; `H` joint absence;
`C` binary association (0 for constant features, rates kept in `r`); `J` joint
observation. If your score is directed, say so in the docstring: `marginal-rate`
(symmetric similarity) and `coverage-marginal` (directed coverage) are distinct
methods for exactly this reason.

## Add a signature

`mechfedgnn/signature_provider.py` — extend `summarize` to return an extra key
under `sig`, and add it to `aggregate_only` **only if a client may transmit it**.
That second step is a privacy decision, not a formatting one: whatever
`aggregate_only` returns is what leaves the client.

Per-row masks, features, labels and per-example predictions must never be added.

## Add a weighting policy

`mechfedgnn/weighting.py`:

```python
@dataclass
class MyPolicy:
    gamma: float = 0.5
    name: str = "my-policy"

    def weights(self, scores, sizes, receiver):
        ...
        return w, float(self.gamma), ""     # (donor weights, self-weight, fallback reason)

WEIGHTING.register("my-policy", lambda gamma=0.5, **kw: MyPolicy(gamma=gamma))
```

`w[receiver]` must be 0 and `w` must sum to 1 — the aggregator enforces both.
Keep `gamma` (self-weight), `p` (sample sizes) and `w` (donor weights) separate.

## Add a learner

`mechfedgnn/learner.py` — implement `initial_state`, `train`, `predict`,
`metrics` and `schema`, then register it. The active learner is the mask-aware
MLP with prediction loss only; a GNN, reconstruction head or attention module is
**out of scope for this phase** and must not be added without a methodology
decision.

If your learner retains optimizer state across calls, say so: the checkpoint
currently records `reset-per-local-training-call` because the present learner
builds a fresh Adam optimizer each call. Persisting optimizer state would change
the algorithm and must be a deliberate, documented choice
(`mechfedgnn/checkpoint.py`).

## Before you replace an existing implementation

Run the golden-reference tests. They assert, against fixtures captured from the
original research code, that assignments, masks, signatures, scores (including
their `None` decisions), weights, gamma, fallback reasons and a small
end-to-end run are unchanged:

```bash
python -m pytest tests/test_golden_equivalence.py -q
```

If you intend to change behaviour, regenerate the fixtures deliberately
(`python -m tools.capture_reference`) in the **same commit**, and say in the
message what changed and why. Silent drift is what these tests exist to stop.
