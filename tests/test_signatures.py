import numpy as np
import pytest

from data.inject import inject, panel_probs_for_rate
from data.validate_injector import phi as injector_phi
from signatures import (chi2, histograms, joint_absence, joint_observation, phi,
                        population_similarity, rates, signature)
from tests.fixtures import X_, Y_, Z_, mask_a, synthetic


def test_fixture_phi_and_joint_absence():
    C, H = phi(mask_a()), joint_absence(mask_a())
    assert C[X_, Z_] == 1.0 and C[Z_, X_] == 1.0
    assert H[X_, Z_] == 0.5
    # Y is always observed -> phi undefined -> its row/col is 0 by convention
    assert np.all(C[Y_, :] == 0) and np.all(C[:, Y_] == 0)
    assert np.all(np.diag(C) == 0)


def test_constant_rule_always_missing_column():
    M = mask_a()
    M[:, Y_] = 0
    C = phi(M)
    assert np.all(C[Y_, :] == 0) and np.all(C[:, Y_] == 0)
    assert rates(M)[Y_] == 1.0          # the rate is kept separately


def test_joint_observation_identity_random_masks():
    rng = np.random.default_rng(1)
    for _ in range(20):
        M = (rng.random((200, 6)) < rng.uniform(0.3, 0.9, 6)).astype(np.int8)
        r, H, J = rates(M), joint_absence(M), joint_observation(M)
        np.testing.assert_allclose(J, 1 - r[:, None] - r[None, :] + H, atol=1e-12)


def test_phi_invariance_under_mask_flip():
    rng = np.random.default_rng(2)
    M = (rng.random((300, 5)) < 0.6).astype(np.int8)
    M[:, 1] = M[:, 0] & (rng.random(300) < 0.8)
    np.testing.assert_allclose(phi(M), phi(1 - M), atol=1e-12)


def test_phi_range_and_chi2():
    rng = np.random.default_rng(3)
    M = (rng.random((250, 6)) < 0.7).astype(np.int8)
    C = phi(M)
    assert np.all(C >= -1 - 1e-12) and np.all(C <= 1 + 1e-12)
    np.testing.assert_allclose(chi2(M), 250 * C**2)


def test_phi_consistent_with_injector_temporary_phi():
    X, y, roles = synthetic(n=2000)
    p = panel_probs_for_rate(2, 0.3, 0.05)
    rng = np.random.default_rng(4)
    masks = [inject(X, y, roles, p, "mar", rng, driver_overlap=0.5)[0],
             inject(X, y, roles, p, "mcar", rng)[0],
             mask_a(),
             (rng.random((100, 4)) < 0.5).astype(np.int8)]
    for M in masks:
        np.testing.assert_allclose(phi(M), injector_phi(M)[0], atol=1e-12)


def test_signature_bundle():
    s = signature(mask_a())
    assert set(s) >= {"n", "r", "H", "J", "C", "chi2"}
    assert s["n"] == 100


# ---------------------------------------------------------------- population S

def test_S_identical_histograms_is_one():
    h = {"u": np.array([0.2, 0.3, 0.5]), "v": np.array([1.0, 0.0])}
    assert population_similarity(h, dict(h)) == pytest.approx(1.0)


def test_S_disjoint_histograms_is_zero():
    assert population_similarity({"u": np.array([1.0, 0.0])},
                                 {"u": np.array([0.0, 1.0])}) == pytest.approx(0.0)


def test_S_characteristic_in_one_client_is_ignored_not_zero():
    hi = {"u": np.array([0.5, 0.5]), "only_i": np.array([1.0, 0.0])}
    hj = {"u": np.array([0.5, 0.5]), "only_j": np.array([0.0, 1.0])}
    assert population_similarity(hi, hj) == pytest.approx(1.0)


def test_S_no_shared_characteristics_is_none():
    assert population_similarity({"u": np.array([1.0])}, {"v": np.array([1.0])}) is None


def test_histograms_normalised_on_shared_edges():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((500, 2))
    edges = {"a": np.quantile(X[:, 0], np.linspace(0, 1, 11)),
             "b": np.quantile(X[:, 1], np.linspace(0, 1, 11))}
    h = histograms(X * 1.5, ["a", "b"], edges)       # values outside the edges are clipped in
    for name in ["a", "b"]:
        assert len(h[name]) == 10
        assert h[name].sum() == pytest.approx(1.0)
