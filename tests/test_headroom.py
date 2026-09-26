import numpy as np

from headroom import headroom_row, verdict

CFG = {"headroom": {"none_max": 0.02, "thin_max": 0.10, "harm_se": 2.0}}


def _ref(local_err, pooled_err):
    return {"local_err": np.asarray(local_err, float), "pooled_err": np.asarray(pooled_err, float),
            "local_best": 100, "local_halt": 1350, "pooled_best": 400, "pooled_halt": 1650}


def test_verdict_thresholds():
    assert verdict(1.0, 0.80) == "HEADROOM OK"
    assert verdict(1.0, 0.95) == "THIN HEADROOM"
    assert verdict(1.0, 0.99) == "NO HEADROOM"
    assert verdict(1.0, 1.05, harm_z=0.5) == "NO HEADROOM"      # worse, but within noise
    assert verdict(1.0, 1.05, harm_z=3.0) == "POOLING HARMS"


def test_pooling_harms_detected_beyond_noise():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(500)
    row = headroom_row(_ref(e, 1.3 * e), CFG)                    # pooled uniformly worse
    assert row["verdict"] == "POOLING HARMS"
    assert row["pooling_harm"] > 0 and row["harm_z"] > 2


def test_negative_headroom_within_noise_is_not_harm():
    # Same error distribution for both references: across many draws the paired
    # test must call harm about as rarely as a 2-SE one-sided test should (~2%).
    rng = np.random.default_rng(1)
    rows = [headroom_row(_ref(rng.standard_normal(60), rng.standard_normal(60)), CFG) for _ in range(400)]
    worse = [r for r in rows if r["pooling_harm"] > 0]
    harms = sum(r["verdict"] == "POOLING HARMS" for r in rows)
    assert len(worse) > 100                       # pooling often looks worse by chance...
    assert harms / len(rows) < 0.05               # ...but is rarely declared harmful


def test_pooling_harm_is_zero_when_pooling_helps():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(300)
    row = headroom_row(_ref(e, 0.5 * e), CFG)
    assert row["pooling_harm"] == 0.0 and row["verdict"] == "HEADROOM OK"
