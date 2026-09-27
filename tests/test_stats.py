import numpy as np
import pandas as pd
import pytest

from loop import arm_weights
from stats import decide, paired, power_table, seed_interval

AC = {"alpha": 1.0, "beta": 1.0, "gamma": 0.5}


def test_decision_rule_against_delta():
    assert decide(-5.0, -3.0, 2.0) == "meaningful (arm better)"
    assert decide(2.5, 4.0, 2.0) == "meaningful (arm worse)"
    assert decide(-1.0, 1.5, 2.0) == "negligible"
    assert decide(-3.0, 1.0, 2.0) == "unresolved"


def test_seed_interval_clusters_receivers_within_seed():
    # receivers inside a seed are averaged first: 3 seeds -> df = 2
    idx = pd.MultiIndex.from_product([[1, 2, 3], ["c0", "c1"]], names=["seed", "receiver"])
    d = pd.Series([1.0, 3.0, 2.0, 2.0, 3.0, 1.0], index=idx)          # every seed mean = 2
    ci = seed_interval(d)
    assert ci["seeds"] == 3 and ci["mean_pct"] == pytest.approx(2.0) and ci["sd_seed"] == pytest.approx(0.0)


def test_paired_relative_difference_is_percent_of_comparator():
    tab = pd.DataFrame({"a": [9.0, 11.0], "b": [10.0, 10.0]})
    np.testing.assert_allclose(paired(tab, "a", "b"), [-10.0, 10.0])


def _metrics(sd, seeds=3):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(seeds):
        shift = rng.normal(0, sd)
        for rec in ["c0", "c1"]:
            for arm in ["local-only", "fedavg", "uniform-donor", "marginal-rate", "missingness-similarity",
                        "coverage-W_H", "coverage-W_C", "population-S", "combined-Q"]:
                v = 1.0 + (shift / 100 if arm == "fedavg" else 0.0)
                rows.append({"seed": s, "receiver": rec, "arm": arm, "fold": "val", "timepoint": "t2",
                             "metric": "rmse", "value": v})
    return pd.DataFrame(rows)


def test_power_flags_noisy_contrasts_unresolvable():
    quiet = power_table(_metrics(0.1), n_study=10).set_index(["arm", "vs"])
    loud = power_table(_metrics(10.0), n_study=10).set_index(["arm", "vs"])
    assert quiet.loc[("fedavg", "local-only"), "resolvable"]
    assert not loud.loc[("fedavg", "local-only"), "resolvable"]


def test_fallback_fires_when_any_donor_score_is_undefined():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    scores = {1: {"s": 0.9}, 2: {"s": None}, 3: {"s": 0.2}}
    w, gamma, why = arm_weights("missingness-similarity", 0, scores, p, AC)
    assert why == "undefined" and gamma == 0.5
    np.testing.assert_allclose(w, [0, 0.2 / 0.9, 0.3 / 0.9, 0.4 / 0.9])      # sample-size weights


def test_fallback_fires_when_scores_do_not_discriminate():
    p = np.array([0.25] * 4)
    scores = {j: {"W_H": 0.7} for j in [1, 2, 3]}
    w, _, why = arm_weights("coverage-W_H", 0, scores, p, AC)
    assert why == "non-discriminating"
    np.testing.assert_allclose(w, [0, 1 / 3, 1 / 3, 1 / 3])


def test_no_fallback_when_scores_discriminate():
    p = np.array([0.25] * 4)
    scores = {1: {"W_H": 0.9}, 2: {"W_H": 0.3}, 3: {"W_H": 0.3}}
    w, _, why = arm_weights("coverage-W_H", 0, scores, p, AC)
    assert why == "" and w[1] == pytest.approx(0.6)
