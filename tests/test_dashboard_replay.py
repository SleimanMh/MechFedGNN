"""The dashboard's replay reader must not alter what the experiments computed.

The research runs never saved realised donor weights, so the dashboard recomputes
them. These tests assert the recomputation is the research code path and not a
re-implementation: it must reproduce, exactly, the weights captured from the
ORIGINAL implementation in tests/golden/reference.json.
"""
import copy
import json
import os

import pandas as pd
import pytest

from dashboard import replay
from data.inject import build_roles
from loop import run_seed
from report import write_run
from tools.capture_reference import SEED, base_cfg, synthetic_df

ARTIFACT_PRECISION = 1e-8   # >> the 1e-10 that %.10g permits, << any meaningful drift

GOLDEN = json.load(open(os.path.join("tests", "golden", "reference.json"), encoding="utf-8"))


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """A real run directory produced by the real orchestrator, then read back."""
    root = str(tmp_path_factory.mktemp("results"))
    df = synthetic_df()
    roles = build_roles(df)
    cfg = base_cfg()
    out = run_seed(df, roles, copy.deepcopy(cfg), SEED, folds=("val", "test"), headroom=False)
    run_dir = write_run("e1", cfg, "synthetic", df, roles, [], [out], root=root)
    return replay.load_run(run_dir), out


def test_recomputed_weights_match_the_golden_reference(replayed):
    run, _ = replayed
    golden = GOLDEN["end_to_end"]["e1"]["weights"]
    seed = run["seeds"][0]

    got = {}
    for arm in {w["arm"] for w in golden}:
        for rec in {w["receiver"] for w in golden}:
            b = replay.aggregation_breakdown(run, seed, rec, arm)
            for d in b["donors"]:
                got[(rec, arm, d["donor"])] = (d["donor_weight"], b["gamma"], b["fallback"] or "")

    assert len(got) == len(golden)
    for w in golden:
        weight, gamma, fallback = got[(w["receiver"], w["arm"], w["donor"])]
        # scores.csv is written with float_format="%.10g", so a weight recomputed
        # from the artifact can only agree to ~1e-10 relative. The next test shows
        # that rounding is the ONLY source of this difference.
        assert weight == pytest.approx(w["weight"], rel=ARTIFACT_PRECISION), w
        assert gamma == pytest.approx(w["gamma"], rel=1e-12), w
        assert fallback == w["fallback"], w


def test_recomputed_weights_match_this_runs_own_realised_weights(replayed):
    """Independent of the fixture: the same run's in-memory realised weights."""
    run, out = replayed
    for w in out["weights"]:
        b = replay.aggregation_breakdown(run, w["seed"], w["receiver"], w["arm"])
        got = {d["donor"]: d["donor_weight"] for d in b["donors"]}
        assert got[w["donor"]] == pytest.approx(float(w["weight"]), rel=ARTIFACT_PRECISION)
        assert b["gamma"] == pytest.approx(float(w["gamma"]), rel=1e-12)


def test_the_only_difference_is_the_artifacts_10_significant_digits(replayed):
    """Feed the SAME reader full-precision scores: the weights then match exactly.

    This is what makes the tolerance above a property of the saved file rather
    than a loosened criterion - replay's arithmetic is bit-identical.
    """
    run, out = replayed
    exact = dict(run)
    exact["_scores"] = pd.DataFrame(out["scores"])          # unrounded, as computed
    for w in out["weights"]:
        b = replay.aggregation_breakdown(exact, w["seed"], w["receiver"], w["arm"])
        got = {d["donor"]: d["donor_weight"] for d in b["donors"]}
        assert got[w["donor"]] == float(w["weight"])        # exact equality
        assert b["gamma"] == float(w["gamma"])
        assert (b["fallback"] or "") == w["fallback"]


def test_contributions_sum_to_one_and_the_receiver_never_donates_to_itself(replayed):
    run, _ = replayed
    seed = run["seeds"][0]
    for arm in run["method"]["arms"]:
        if arm == "local-only":
            continue
        for c in run["clients"]:
            b = replay.aggregation_breakdown(run, seed, c["id"], arm)
            assert b["total_ok"], (arm, c["id"], b["total_contribution"])
            assert c["id"] not in {d["donor"] for d in b["donors"]}
            assert b["donor_weight_sum"] == pytest.approx(1.0, abs=1e-12)


def test_weight_matrix_rows_sum_to_one_and_the_diagonal_is_gamma(replayed):
    run, _ = replayed
    seed = run["seeds"][0]
    m = replay.weight_matrix(run, seed, "combined-Q")
    ids = m["clients"]
    for r, row in enumerate(m["matrix"]):
        assert sum(row) == pytest.approx(1.0, abs=1e-12)
        b = replay.aggregation_breakdown(run, seed, ids[r], "combined-Q")
        assert row[r] == pytest.approx(b["gamma"], abs=1e-12)


def test_fedavg_self_weight_is_its_own_sample_share_not_the_configured_gamma(replayed):
    """FedAvg's gamma is p_i by definition; it must not be overwritten by cfg gamma."""
    run, _ = replayed
    seed = run["seeds"][0]
    ids, p = replay._sample_shares(run)
    for i, cid in enumerate(ids):
        b = replay.aggregation_breakdown(run, seed, cid, "fedavg")
        assert b["gamma"] == pytest.approx(float(p[i]), rel=1e-12)


def test_scoreless_arms_are_reported_as_using_no_score(replayed):
    run, _ = replayed
    seed = run["seeds"][0]
    for arm in replay.SCORELESS:
        if arm not in run["method"]["arms"]:
            continue
        b = replay.aggregation_breakdown(run, seed, run["clients"][0]["id"], arm)
        assert b["uses_score"] is False
        assert all(d["score"] is None for d in b["donors"])


def test_unavailable_information_is_declared_rather_than_approximated(replayed):
    """The audit found these have no saved source. They must be reported missing."""
    run, _ = replayed
    a = run["availability"]
    for key in ("event_log", "error_sum_and_count", "training_loss_curve",
                "val_test_row_counts", "federated_rounds"):
        assert a[key]["available"] is False
        assert a[key]["note"]
    assert a["saved_donor_weights"]["available"] is False     # research runs never wrote it
    for c in run["clients"]:
        assert c["n_val"] is None and c["n_test"] is None      # not invented
        assert c["n_train"] is not None


def test_a_fallback_is_explained_when_one_occurs(replayed):
    run, _ = replayed
    seed = run["seeds"][0]
    seen = False
    for arm in run["method"]["arms"]:
        for c in run["clients"]:
            b = replay.aggregation_breakdown(run, seed, c["id"], arm)
            if b["fallback"]:
                seen = True
                assert b["fallback_explained"]
                assert "sample-size weighting" in b["fallback_explained"]
    if not seen:
        pytest.skip("no fallback occurred in this reference run")


def test_load_run_reads_real_result_directories_if_present():
    """Smoke test against the actual research results, when they exist locally."""
    runs = replay.list_runs()
    if not runs:
        pytest.skip("no results/ directory in this checkout")
    r = replay.load_run(runs[0]["path"])
    assert r["clients"] and r["seeds"]
    b = replay.aggregation_breakdown(r, r["seeds"][0], r["clients"][0]["id"],
                                     r["method"]["arms"][-1])
    assert b["total_ok"]
