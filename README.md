# MechFedGNN

Personalised federated learning for tabular data where clients are incomplete in
different ways. Each client summarises its own missingness mask and its own
population; a server turns those summaries into directed, personalised
aggregation weights. This repo tests whether those summaries predict useful
collaboration.

## Getting started

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Then open Claude Code in this folder and give it:

> Read CLAUDE.md in full. Do Stage 0 only: write `data/download_uci.py`, fetch
> concrete, housing and wine, then write `data/inject.py` (panel-based, §3) and
> `data/validate_injector.py` (§3.5) and run validation on all three datasets.
> Stop and show me the panel assignment per dataset, the within-panel vs
> cross-panel phi table, the null and dose-response verdicts, and the
> missing-panels-per-row table. Don't build anything else yet.

## Layout

- `CLAUDE.md` — the implementation plan. Authoritative; read it before writing code.
- `.claude/skills/run-report/` — the report every experiment run must produce.
- `data/`, `configs/`, `results/`, `tests/` — created as the stages proceed.

## Stages

| Stage | Output | Stop and review |
|---|---|---|
| 0 | data download + panel injector + validation | panel assignment, phi tables, null/dose verdicts |
| 1 | signatures, scores, kernel + tests | full pytest output |
| 2 | clients, model, loop, headroom, report | headroom verdicts, frozen budget |
| 3 | E1, E2, E3 | the E1 report, before running E2 |
