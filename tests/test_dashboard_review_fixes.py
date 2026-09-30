"""Regression tests for the dashboard defects found in review.

Each test fails against the behaviour that was reported:

  1. the aggregation explanation displayed p_j under the label for base_j;
  2. replay used first-seed client sizes for every selected seed, silently;
  3. an aggregated model being COMPUTED was animated as if it had been sent,
     and a failed parent request was logged as "model sent";
  4. a live run reported no prediction error at all.
"""
import copy
import math
import os

import numpy as np
import pytest

from dashboard import live, replay
from data.inject import build_roles
from loop import run_seed
from report import write_run
from tools.capture_reference import SEED, base_cfg, synthetic_df


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("results"))
    df = synthetic_df()
    roles = build_roles(df)
    cfg = base_cfg()
    outs = [run_seed(df, roles, copy.deepcopy(cfg), sd, folds=("val", "test"), headroom=False)
            for sd in (SEED, SEED + 12)]
    d = write_run("e1", cfg, "synthetic", df, roles, [], outs, root=root)
    return replay.load_run(d.replace("\\", "/"))


def step(b, n):
    return next(s for s in b["steps"] if s["n"] == n)


# ------------------------------------------------- 1. intermediate values

def test_the_blend_step_shows_base_j_not_the_sample_share(run):
    """With alpha = 1, base_j = q_j. The step used to display p_j instead."""
    seed, rec = run["seeds"][0], run["clients"][0]["id"]
    b = replay.aggregation_breakdown(run, seed, rec, "combined-Q")
    assert b["effective_alpha"] == 1.0 and not b["fallback"]

    scores = {d["donor"]: d["score"] for d in b["donors"]}
    shares = {d["donor"]: d["sample_share"] for d in b["donors"]}
    shown = step(b, 2)["values"]

    assert shown == pytest.approx(scores)            # base_j == q_j when alpha = 1
    differing = [k for k in shown if abs(scores[k] - shares[k]) > 1e-9]
    assert differing, "this run cannot distinguish the two; the test would be vacuous"
    for k in differing:
        assert shown[k] != pytest.approx(shares[k]), f"{k} is still showing p_j"


def test_the_blend_step_matches_the_kernels_formula_for_every_alpha():
    """base_j = alpha*q_j + (1-alpha)*p_j, checked directly against the kernel."""
    from kernel import donor_weights
    q = [None, 0.8, 0.4, 0.6]
    p = np.array([0.25, 0.25, 0.30, 0.20])
    for alpha in (0.0, 0.25, 0.5, 1.0):
        base = [p[j] if q[j] is None else alpha * q[j] + (1 - alpha) * p[j]
                for j in range(len(p))]
        expect = np.array(base, float)
        expect[0] = 0.0
        np.testing.assert_allclose(donor_weights(0, q, p, alpha=alpha, beta=1.0),
                                   expect / expect.sum(), rtol=1e-12)
    # the specific case from the review: alpha=1, q=0.8, p=0.25 -> base = 0.8
    assert 1.0 * 0.8 + 0.0 * 0.25 == 0.8


def test_fedavg_is_explained_with_its_own_pinned_alpha_and_self_weight(run):
    """fedavg pins alpha = 0 and gamma = p_i regardless of the configuration."""
    seed, rec = run["seeds"][0], run["clients"][0]["id"]
    b = replay.aggregation_breakdown(run, seed, rec, "fedavg")
    assert (b["effective_alpha"], b["effective_beta"]) == (0.0, 1.0)
    ids, p = replay._sample_shares(run)
    assert b["gamma"] == pytest.approx(float(p[ids.index(rec)]))
    shares = {d["donor"]: d["sample_share"] for d in b["donors"]}
    assert step(b, 2)["values"] == pytest.approx(shares)      # here base_j IS p_j
    assert "alpha = 0 by definition" in step(b, 2)["detail"]
    assert "sample share p_i" in step(b, 5)["detail"]


def test_uniform_donor_is_explained_as_using_neither_score_nor_size(run):
    seed, rec = run["seeds"][0], run["clients"][0]["id"]
    b = replay.aggregation_breakdown(run, seed, rec, "uniform-donor")
    for n in (1, 2, 3):
        assert step(b, n)["inactive"] is True
        assert step(b, n)["values"] == {}
    k = len(b["donors"])
    assert all(d["donor_weight"] == pytest.approx(1.0 / k) for d in b["donors"])


def test_local_only_is_explained_as_no_aggregation_at_all(run):
    seed, rec = run["seeds"][0], run["clients"][0]["id"]
    b = replay.aggregation_breakdown(run, seed, rec, "local-only")
    assert b["gamma"] == 1.0
    assert all(d["donor_weight"] is None for d in b["donors"])   # nothing is mixed in
    assert all(d["effective_contribution"] is None for d in b["donors"])
    assert len(b["steps"]) == 1 and b["steps"][0]["inactive"] is True


def test_a_fallback_is_explained_as_ignoring_the_score():
    """Under the declared fallback the weighting runs with alpha = 0, so the
    explanation must show p_j and say the score was not used."""
    ac = {"alpha": 1.0, "beta": 1.0, "gamma": 0.5}
    rows = [{"donor": "c1", "score": None, "sample_share": 0.4, "donor_weight": 0.4,
             "effective_contribution": 0.2},
            {"donor": "c2", "score": None, "sample_share": 0.6, "donor_weight": 0.6,
             "effective_contribution": 0.3}]
    steps = replay._steps("combined-Q", "Q", rows, np.array([0.0, 0.4, 0.6]), 0, ac, 0.5,
                          "undefined")
    s2 = next(s for s in steps if s["n"] == 2)
    assert s2["values"] == {"c1": 0.4, "c2": 0.6}              # p_j, because alpha is forced to 0
    assert "fallback" in s2["detail"] and "not used" in s2["detail"]
    assert next(s for s in steps if s["n"] == 1)["inactive"] is True


def test_every_displayed_step_value_is_finite_and_named(run):
    seed = run["seeds"][0]
    for arm in run["method"]["arms"]:
        for c in run["clients"]:
            b = replay.aggregation_breakdown(run, seed, c["id"], arm)
            for s in b["steps"]:
                for k, v in (s["values"] or {}).items():
                    assert v is None or math.isfinite(float(v)), (arm, s["n"], k)


# ------------------------------------------------- 2. per-seed provenance

def test_client_sizes_are_declared_as_first_seed_only(run):
    assert run["params_seed"] == min(run["seeds"])
    a = run["availability"]["per_seed_client_sizes"]
    assert a["available"] is False and "FIRST" in a["note"]


def test_size_dependent_arms_are_flagged_on_a_non_first_seed(run):
    """fedavg weights depend on client sizes, which were saved for one seed."""
    if len(run["seeds"]) < 2:
        pytest.skip("single-seed run")
    other = run["seeds"][1]
    b = replay.aggregation_breakdown(run, other, run["clients"][0]["id"], "fedavg")
    assert b["weights_use_client_sizes"] is True
    assert b["weights_exact_for_this_seed"] is False
    assert b["seed_caveat"] and str(run["params_seed"]) in b["seed_caveat"]


def test_score_only_arms_are_exact_on_any_seed(run):
    """With alpha = 1 and no fallback the weights come entirely from that
    seed's saved scores, so there is nothing to caveat."""
    for seed in run["seeds"]:
        b = replay.aggregation_breakdown(run, seed, run["clients"][0]["id"], "combined-Q")
        assert b["weights_use_client_sizes"] is False
        assert b["weights_exact_for_this_seed"] is True
        assert b["seed_caveat"] is None


def test_size_dependence_is_decided_per_arm_not_guessed():
    assert replay.size_dependent("fedavg", {"alpha": 1.0}, "") is True
    assert replay.size_dependent("combined-Q", {"alpha": 1.0}, "") is False
    assert replay.size_dependent("combined-Q", {"alpha": 1.0}, "undefined") is True
    assert replay.size_dependent("combined-Q", {"alpha": 0.5}, "") is True
    assert replay.size_dependent("uniform-donor", {"alpha": 1.0}, "") is False
    assert replay.size_dependent("local-only", {"alpha": 0.0}, "") is False


def test_the_weight_matrix_carries_the_same_caveat(run):
    if len(run["seeds"]) < 2:
        pytest.skip("single-seed run")
    m = replay.weight_matrix(run, run["seeds"][1], "fedavg")
    assert m["weights_exact_for_this_seed"] is False and m["seed_caveat"]
    m2 = replay.weight_matrix(run, run["seeds"][1], "combined-Q")
    assert m2["weights_exact_for_this_seed"] is True and m2["seed_caveat"] is None


# ------------------------------------------------- 3. transfer semantics

def _observed(monkeypatch, bus, status, route):
    monkeypatch.setattr(live.ServerApp, "handle",
                        lambda self, r, b, i: (status, b'{"ok": false, "error": "refused"}'))
    obs = live.ObservedApp.__new__(live.ObservedApp)
    obs.bus = bus
    obs.handle(route, b"", "c0")
    return bus.since(0)


def test_a_failed_parent_request_is_not_reported_as_a_model_sent(monkeypatch):
    """Regression: every /v1/parent call was logged as 'parent model sent'."""
    bus = live.EventBus()
    events = _observed(monkeypatch, bus, 409, "/v1/parent")
    assert [e["kind"] for e in events] == ["reject"]
    assert "no model was sent" in events[0]["text"]
    assert "sent the response" not in events[0]["text"]


def test_a_successful_parent_request_is_a_delivery_and_claims_no_receipt(monkeypatch):
    bus = live.EventBus()
    monkeypatch.setattr(live.ServerApp, "handle", lambda self, r, b, i: (200, b"payload"))
    obs = live.ObservedApp.__new__(live.ObservedApp)
    obs.bus = bus
    obs.handle("/v1/parent", b"", "c0")
    (e,) = bus.since(0)
    assert e["kind"] == "delivered" and e["observed"] == "request_and_response"
    assert "requested" in e["text"] and "not claimed" in e["text"]


def test_computing_a_model_is_not_a_transfer_event():
    """The server aggregating a model says nothing about any client receiving
    it, so it must not be emitted as a transfer."""
    bus = live.EventBus()
    bus.emit("computed", "server computed the personalised model for c0", client="c0")
    (e,) = bus.since(0)
    assert e["kind"] == "computed"
    assert "direction" not in e          # nothing for the UI to animate as a packet


def test_the_frontend_animates_only_observed_transfers():
    js = open(os.path.join("dashboard", "static", "live.js"), encoding="utf-8").read()
    assert 'e.kind === "transfer" || e.kind === "delivered"' in js
    assert 'e.kind === "computed") pulseServer()' in js
    assert '"aggregate"' not in js       # the conflated event type is gone


def test_unobserved_receipt_is_declared_unavailable():
    api = object()
    r = live.routes(api, allow_launch=False)["live/status"](api, {})
    assert "client_confirmed_receipt" in r["unavailable"]


# ------------------------------------------------- 4. live evaluation

def test_rmse_is_derived_from_counts_and_summed_squared_error_only():
    table = live.rmse_table({1: {
        "c0": {"received": {"fold": "val", "n": 40, "sse": 160.0},
               "local": {"fold": "val", "n": 40, "sse": 40.0}},
        "c1": {"received": {"fold": "val", "n": 60, "sse": 240.0},
               "local": {"fold": "val", "n": 60, "sse": 60.0}}}})
    by = {r["client"]: r for r in table}
    assert by["c0"]["local_rmse"] == pytest.approx(1.0)        # sqrt(40/40)
    assert by["c1"]["received_rmse"] == pytest.approx(2.0)     # sqrt(240/60)
    micro = by["ALL (micro-average)"]
    assert micro["local_n"] == 100
    assert micro["local_rmse"] == pytest.approx(math.sqrt(100.0 / 100))
    assert micro["received_rmse"] == pytest.approx(math.sqrt(400.0 / 100))


def test_an_empty_fold_does_not_produce_a_fake_zero():
    assert math.isnan(live._rmse({"n": 0, "sse": 0.0}))


def test_a_client_evaluation_contains_no_labels_or_predictions(tmp_path):
    """What the client sends upward is two numbers per model, nothing else."""
    from mechfedgnn.client_runtime import ClientRuntime, save_client_shard
    rng = np.random.default_rng(0)
    X, M = rng.standard_normal((40, 4)), np.ones((40, 4), np.int8)
    y = X[:, 0] + 0.1 * rng.standard_normal(40)
    folds = {"train": np.arange(24), "val": np.arange(24, 32), "test": np.arange(32, 40)}
    p = save_client_shard(str(tmp_path / "c0.npz"), X, M, y, folds,
                          {"always_observed": [0], "maskable": [1, 2, 3]}, "c0")
    rt = ClientRuntime(client_id="c0", experiment_id="demo", shard_path=p, transport=None)
    rt.load()
    state = rt.learner.initial_state(4, seed=0)

    ev = rt.evaluate(state, "val")
    assert set(ev) == {"fold", "n", "sse"}
    assert ev["n"] == 8 and ev["sse"] > 0 and math.isfinite(ev["sse"])
    for banned in ("y", "pred", "predictions", "labels", "X", "M"):
        assert banned not in ev

    # and it really is the RMSE the research metric would report
    xin = rt.transform.inputs(X[folds["val"]], M[folds["val"]])
    pred = rt.learner.predict(state, 4, xin, rt.transform)
    expected = float(np.sqrt(((pred - y[folds["val"]]) ** 2).mean()))
    assert math.sqrt(ev["sse"] / ev["n"]) == pytest.approx(expected, rel=1e-12)


def test_the_evaluate_fold_option_is_validated_like_every_other():
    assert live.validate({"evaluate_fold": "test"})["evaluate_fold"] == "test"
    assert live.validate({})["evaluate_fold"] == "val"
    with pytest.raises(ValueError, match="evaluate_fold"):
        live.validate({"evaluate_fold": "train"})
    with pytest.raises(ValueError, match="evaluate_fold"):
        live.validate({"evaluate_fold": "../../etc/passwd"})


@pytest.mark.slow
def test_a_real_live_run_reports_prediction_error_from_real_clients():
    import time
    bus = live.EventBus()
    s = live.Session(bus=bus)
    s.start({"clients": 2, "rounds": 1, "local_steps": 5, "seed": 3,
             "arm": "combined-Q", "evaluate_fold": "val"})
    deadline = time.time() + 240
    while time.time() < deadline and s.state in ("idle", "preparing", "running"):
        time.sleep(0.25)
    assert s.state == "finished", s.error
    s._thread.join(60)

    rows = live.rmse_table(s.evaluations)
    assert rows, "no client reported any evaluation"
    for r in rows:
        assert r["local_n"] > 0
        assert math.isfinite(r["local_rmse"]) and r["local_rmse"] > 0
        assert r["local_rmse"] == pytest.approx(math.sqrt(r["local_sse"] / r["local_n"]))
    assert any(r["client"].startswith("ALL") for r in rows)

    kinds = {e["kind"] for e in bus.since(0)}
    assert "evaluation" in kinds and "computed" in kinds and "delivered" in kinds
    assert "aggregate" not in kinds          # the conflated event type is gone
