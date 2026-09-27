import pandas as pd

from loop import ARMS
from report import averaging_harm


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
