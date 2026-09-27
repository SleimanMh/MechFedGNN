import pandas as pd

from loop import ARMS
from report_tables import averaging_harm


def _metrics():
    """2 seeds x 2 receivers. fedavg worse than local-only on c1 in both seeds
    (harm), better on c0; every other arm equals local-only (no harm)."""
    rows = []
    for seed in [1, 2]:
        for rec, fed in [("c0", 0.8), ("c1", 1.3)]:
            for arm in ARMS:
                v = fed if arm == "fedavg" else 1.0
                for tp in ["t1", "t2"]:
                    rows.append({"seed": seed, "receiver": rec, "arm": arm, "timepoint": tp,
                                 "metric": "rmse", "value": v})
    return pd.DataFrame(rows)


def test_averaging_harm_signs_and_counts():
    text = "\n".join(averaging_harm(_metrics()))
    assert "| c0 | 0.2 | 0 | 0.2 | 0 |" in text          # fedavg better on c0: positive
    assert "| c1 | -0.3 | 0 | -0.3 | 0 |" in text        # fedavg worse on c1: negative
    assert "| fedavg | 1/2 | 2/4 | 1/2 | 2/4 |" in text
    assert "| uniform-donor | 0/2 | 0/4 | 0/2 | 0/4 |" in text


def test_report_end_to_end_on_synthetic_data(tmp_path):
    """Six sections, plus the E1 companion tables, from a tiny two-seed run."""
    import copy

    import yaml

    from data.inject import build_roles
    from loop import run_seed
    from report import write_run
    from tests.test_deferred import _synthetic_df

    df = _synthetic_df(900)
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    cfg["model"]["local_steps"], cfg["model"]["adapt_budget"] = 30, 5
    cfg["headroom"]["early_stopping"].update(max_steps=200, patience=2)
    cfg["seeds"] = [11, 23]
    roles = build_roles(df)
    outs = [run_seed(df, roles, copy.deepcopy(cfg), s, folds=("val", "test")) for s in cfg["seeds"]]
    out = write_run("t", cfg, "synthetic", df, roles, [], outs, root=str(tmp_path))
    text = open(f"{out}/REPORT.md", encoding="utf-8").read()
    heads = [line for line in text.splitlines() if line.startswith("## ")]
    assert [h.split(".")[0] for h in heads] == ["## 1", "## 2", "## 3", "## 4", "## 5", "## 6"]
    for marker in ["Paired contrasts", "Validation-selected candidate", "Group recovery", "Compute",
                   "Harm from averaging", "Declared fallback"]:
        assert marker in text, marker
