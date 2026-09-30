"""Milestone A: the extracted interfaces must reproduce the reference behaviour.

The fixtures in tests/golden/reference.json were captured from the ORIGINAL
implementation (tools/capture_reference.py) before extraction. These tests are
what protects every later replacement: if a component is reimplemented and any
of these drift, the refactor changed behaviour.

Numerical policy: exact for indices, masks and signatures; atol 1e-8 for
parameters and weights; bitwise equality across different hardware is NOT
required.
"""
import copy
import json
import os

import numpy as np
import pytest

from data.clients import build_clients
from data.e5 import build_e5_clients
from data.inject import build_roles
from data.matched import build_matched_clients
from mechfedgnn.aggregation import WeightedAverage
from mechfedgnn.registry import SCORES
from mechfedgnn.scoring import SCORELESS_ARMS
from mechfedgnn.signature_provider import MaskSignatureProvider
from mechfedgnn.weighting import policy_for
from tools.capture_reference import base_cfg, synthetic_df

GOLDEN = os.path.join("tests", "golden", "reference.json")
ATOL = 1e-8

pytestmark = pytest.mark.skipif(not os.path.exists(GOLDEN),
                                reason="run `python -m tools.capture_reference` first")


@pytest.fixture(scope="module")
def ref():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def setup():
    df = synthetic_df()
    return df, build_roles(df), base_cfg()


def _builders(df, roles, cfg):
    cfg_e5 = copy.deepcopy(cfg)
    cfg_e5["e5"] = {"P": 1, "M": 1, "condition": "P1M1", "s_config": "primary", "s_exclude": ()}
    return {"e1": lambda: build_clients(df, roles, cfg, cfg["seeds"][0]),
            "e1m_b": lambda: build_matched_clients(df, roles, cfg, cfg["seeds"][0], "b")[:2],
            "e5_P1M1": lambda: build_e5_clients(df, roles, cfg_e5, cfg["seeds"][0])[:2]}


@pytest.mark.parametrize("builder", ["e1", "e1m_b", "e5_P1M1"])
def test_assignments_and_masks_are_unchanged(ref, setup, builder):
    df, roles, cfg = setup
    clients, _ = _builders(df, roles, cfg)[builder]()
    for c, g in zip(clients, ref["builders"][builder]["clients"]):
        assert c["id"] == g["id"] and len(c["rows"]) == g["n_rows"]
        for fold in ["train", "val", "test"]:
            np.testing.assert_array_equal(c[fold], np.array(g[fold]))
        np.testing.assert_array_equal((c["M"] == 0).sum(0), np.array(g["mask_col_missing"]))


@pytest.mark.parametrize("builder", ["e1", "e1m_b", "e5_P1M1"])
def test_signatures_are_unchanged(ref, setup, builder):
    df, roles, cfg = setup
    clients, _ = _builders(df, roles, cfg)[builder]()
    provider = MaskSignatureProvider()
    for c in clients:
        s = provider.summarize(c, roles)
        g = ref["builders"][builder]["signatures"][c["id"]]
        assert s["n_train"] == g["n_train"]
        for key in ["r", "H", "J", "C"]:
            np.testing.assert_allclose(s["sig"][key], np.array(g[key]), atol=0, rtol=0)
        assert sorted(s["hist"]) == sorted(g["hist"])
        for k in s["hist"]:
            np.testing.assert_allclose(s["hist"][k], np.array(g["hist"][k]), atol=0, rtol=0)


@pytest.mark.parametrize("builder", ["e1", "e1m_b", "e5_P1M1"])
def test_scores_and_none_decisions_are_unchanged(ref, setup, builder):
    """Every registered score, including which pairs are undefined."""
    df, roles, cfg = setup
    clients, _ = _builders(df, roles, cfg)[builder]()
    provider = MaskSignatureProvider()
    summ = [provider.summarize(c, roles) for c in clients]
    lam = cfg["aggregation"]["lambda_pop"]
    key = {"coverage-W_H": "W_H", "coverage-W_C": "W_C", "coverage-marginal": "W_marg",
           "missingness-similarity": "s", "marginal-rate": "rate", "population-S": "S",
           "combined-Q": "Q", "combined-Q-marginal": "Q_marg"}
    checked = 0
    for i, ci in enumerate(clients):
        for j, cj in enumerate(clients):
            if i == j:
                continue
            g = ref["builders"][builder]["scores"][ci["id"]][cj["id"]]
            for name, gk in key.items():
                got = SCORES.create(name, lambda_pop=lam).score(summ[i], summ[j])
                want = g[gk]
                assert (got is None) == (want is None), f"{name}: None decision changed"
                if got is not None:
                    assert got == pytest.approx(want, abs=ATOL), name
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("builder", ["e1", "e1m_b", "e5_P1M1"])
@pytest.mark.parametrize("tag", ["equal", "nonuniform"])
def test_weights_gamma_and_fallback_are_unchanged(ref, setup, builder, tag):
    """Includes a NON-UNIFORM size fixture: with equal clients every policy
    produces near-identical weights, so a bug could hide behind uniformity."""
    df, roles, cfg = setup
    clients, _ = _builders(df, roles, cfg)[builder]()
    provider = MaskSignatureProvider()
    summ = [provider.summarize(c, roles) for c in clients]
    lam, ac = cfg["aggregation"]["lambda_pop"], cfg["aggregation"]
    k = len(clients)
    sizes = np.full(k, 1.0 / k) if tag == "equal" else np.array([0.10, 0.20, 0.30, 0.40])[:k]
    sizes = sizes / sizes.sum()
    key = {"coverage-W_H": "W_H", "coverage-W_C": "W_C", "coverage-marginal": "W_marg",
           "missingness-similarity": "s", "marginal-rate": "rate", "population-S": "S",
           "combined-Q": "Q", "combined-Q-marginal": "Q_marg"}
    for i, ci in enumerate(clients):
        for arm, g in ((a, ref["builders"][builder]["weights"].get(f"{ci['id']}|{tag}|{a}"))
                       for a in list(SCORELESS_ARMS) + list(key)):
            if g is None:
                continue
            if arm in SCORELESS_ARMS:
                scores = {}
            else:
                strat = SCORES.create(arm, lambda_pop=lam)
                scores = {j: strat.score(summ[i], summ[j]) for j in range(k) if j != i}
            w, gamma, fb = policy_for(arm, ac).weights(scores, sizes, i)
            assert fb == g["fallback"], f"{arm}: fallback reason changed"
            assert gamma == pytest.approx(g["gamma"], abs=ATOL)
            if g["w"] is None:
                assert w is None
            else:
                np.testing.assert_allclose(w, np.array(g["w"]), atol=ATOL)


def test_aggregation_with_nonuniform_weights_is_unchanged(ref):
    g = ref["aggregation"]
    thetas = [{k: np.array(v) for k, v in t.items()} for t in g["thetas"]]
    out = WeightedAverage().combine(thetas, 0, np.array(g["w"]), g["gamma"])
    for k, v in g["mixed"].items():
        np.testing.assert_allclose(out[k], np.array(v), atol=ATOL)
    fed = WeightedAverage().combine(thetas, 1, np.array(g["fedavg_w"]), float(g["fedavg_p"][1]))
    for k, v in g["fedavg_mixed"].items():
        np.testing.assert_allclose(fed[k], np.array(v), atol=ATOL)


def test_aggregator_rejects_incompatible_and_invalid_weights():
    a = {"w": np.zeros(3)}
    b = {"w": np.zeros(4)}
    agg = WeightedAverage()
    with pytest.raises(Exception):
        agg.combine([a, b], 0, np.array([0.0, 1.0]), 0.5)
    with pytest.raises(ValueError):
        agg.combine([a, a], 0, np.array([0.5, 0.5]), 0.5)      # receiver weight nonzero
    with pytest.raises(ValueError):
        agg.combine([a, a], 0, np.array([0.0, 0.7]), 0.5)      # does not sum to 1


@pytest.mark.parametrize("case", ["e1", "e5_P1M1"])
def test_end_to_end_metrics_and_weights_are_unchanged(ref, setup, case):
    """Milestone A acceptance: a small end-to-end run reproduces the reference
    metrics and the REALISED aggregation weights, without rerunning the
    research suite. This is the regression guard for replacing the orchestrator."""
    from loop import run_seed

    df, roles, cfg = setup
    seed = cfg["seeds"][0]
    if case == "e1":
        c, builder = copy.deepcopy(cfg), None
    else:
        c = copy.deepcopy(cfg)
        c["e5"] = {"P": 1, "M": 1, "condition": "P1M1", "s_config": "primary", "s_exclude": ()}
        from mechfedgnn.scoring import SCORELESS_ARMS  # noqa: F401
        from loop import E5_ARMS
        c["arms"] = E5_ARMS
        builder = lambda s: build_e5_clients(df, roles, c, s)[:2]
    out = run_seed(df, roles, c, seed, folds=("val", "test"), headroom=False, builder=builder)

    g = ref["end_to_end"][case]
    got = sorted([{"receiver": r["receiver"], "arm": r["arm"], "fold": r["fold"],
                   "budget": int(r["budget"]), "metric": r["metric"], "value": float(r["value"])}
                  for r in out["metrics"]],
                 key=lambda x: (x["receiver"], x["arm"], x["fold"], x["budget"], x["metric"]))
    want = sorted(g["metrics"],
                  key=lambda x: (x["receiver"], x["arm"], x["fold"], x["budget"], x["metric"]))
    assert len(got) == g["n_rows"] == len(want)
    for a, b in zip(got, want):
        assert (a["receiver"], a["arm"], a["fold"], a["budget"], a["metric"]) == \
               (b["receiver"], b["arm"], b["fold"], b["budget"], b["metric"])
        if np.isnan(b["value"]):
            assert np.isnan(a["value"])
        else:
            assert a["value"] == pytest.approx(b["value"], rel=1e-6, abs=1e-9)

    gw = sorted(g["weights"], key=lambda x: (x["receiver"], x["arm"], x["donor"]))
    ow = sorted([{"receiver": w["receiver"], "arm": w["arm"], "donor": w["donor"],
                  "weight": float(w["weight"]), "gamma": float(w["gamma"]),
                  "fallback": w["fallback"]} for w in out["weights"]],
                key=lambda x: (x["receiver"], x["arm"], x["donor"]))
    assert len(ow) == len(gw)
    for a, b in zip(ow, gw):
        assert (a["receiver"], a["arm"], a["donor"], a["fallback"]) == \
               (b["receiver"], b["arm"], b["donor"], b["fallback"])
        assert a["weight"] == pytest.approx(b["weight"], abs=ATOL)
        assert a["gamma"] == pytest.approx(b["gamma"], abs=ATOL)
