# Historical commands — compatibility

**No historical command changed.** Every research entry point keeps its name,
its flags and its behaviour. The refactor added components alongside them; it
did not rewrite the experiment scripts.

| Command | Status | Notes |
|---|---|---|
| `python run_e1.py [--datasets ...] [--corrected] [--assignment corr\|random]` | unchanged | E1 signal comparison |
| `python run_e1m.py [--datasets ...] [--conditions a b c] [--corrected] [--assignment ...]` | unchanged | E1M matched-marginal |
| `python run_e5.py --stage dev\|eval [--datasets ...] [--conditions ...] [--s-config ...]` | unchanged | E5 population compatibility |
| `python pilots.py [--headroom-only\|--power]` | unchanged | budget / headroom / power pilots |
| `python check_e1m.py` | unchanged | keeps its **cross-client matched-rate tolerance** |
| `python check_e5.py [--datasets ...] [--seeds ...]` | unchanged | keeps its **M0↔M1 per-client count invariant** |
| `python -m analysis.*` | unchanged | every analysis script |
| `python -m data.validate_injector` | unchanged | injector validation |

## Why the two construction checks stay different

E1M matches marginal rates across clients **by design**, so a cross-client rate
tolerance is the right check there. E5 deliberately gives clients **different**
rates, so the same check would forbid the very heterogeneity the experiment
needs; E5's invariant is instead that M0 and M1 have identical per-feature
counts per client and fold.

Applying one check globally was a mistake made once and corrected before any
E5 prediction result existed (`docs/E5_DIAGNOSTIC.md` §4). The framework keeps
them per-experiment and must not unify them.

## Results are never rewritten

Existing directories under `results/` are untouched. New runs write to new
directories, and the artifact store **refuses to overwrite a completed run** —
it raises `RunExists` and asks for a new run id.

## Reproducing the historical experiments

```bash
python -m data.download_uci                     # fetch raw tables (gitignored)
python check_e5.py                              # construction checks
python run_e1.py  --corrected --assignment corr
python run_e1m.py --corrected --assignment corr
python run_e5.py  --stage eval
python -m analysis.build_results_md             # rebuild docs/RESULTS.md
```

`raw/` and `results/` are gitignored, so a fresh clone has no data until
`download_uci` runs. The golden-reference tests deliberately use **synthetic**
data so they pass on a clone with no downloads.

## New, optional entry points

These are additions; nothing depends on them.

| Command | Purpose |
|---|---|
| `python -m tools.capture_reference` | regenerate golden fixtures from the current code |
| `python -m demo.prepare_shards` | build one data shard per client |
| `python -m demo.server --rounds N [--tls DIR]` | coordinator process |
| `python -m demo.client --client-id cK --url URL [--tls DIR]` | one client process |

## Using the framework from Python

```python
from mechfedgnn.config import load
from mechfedgnn.runner import execute

cfg = load("configs/defaults.yaml", dataset="wine")     # validated + resolved
run_dir, out = execute("my_experiment", "wine", df, roles, cfg, seed=11)
```

`execute` writes provenance (revision, dirty flag, checksums, split/mask ids,
seed streams, preprocessing and where it was fitted), the metrics, the scores,
the **realised** aggregation weights, fallback reasons and participation.
