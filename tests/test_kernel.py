import numpy as np
import pytest

from kernel import aggregate, donor_weights


def test_alpha_one_weights_follow_scores():
    # clients [A, B, C]; receiver A; q = W_H(A<-.) from the fixture
    w = donor_weights(0, q=[None, 0.9, 0.3], p=[1 / 3] * 3, alpha=1.0, beta=1.0)
    np.testing.assert_allclose(w, [0.0, 0.75, 0.25])


def test_aggregate_fixture_mix():
    rng = np.random.default_rng(0)
    th = [rng.standard_normal(7) for _ in range(3)]
    w = donor_weights(0, q=[None, 0.9, 0.3], p=[1 / 3] * 3, alpha=1.0, beta=1.0)
    out = aggregate(th, 0, w, gamma=0.5)
    np.testing.assert_allclose(out, 0.5 * th[0] + 0.375 * th[1] + 0.125 * th[2], atol=1e-12)


def test_aggregate_accepts_parameter_dicts():
    th = [{"a": np.full(2, float(k)), "b": np.full((2, 2), 10.0 * k)} for k in range(3)]
    out = aggregate(th, 0, np.array([0.0, 0.5, 0.5]), gamma=0.5)
    np.testing.assert_allclose(out["a"], 0.75)
    np.testing.assert_allclose(out["b"], 7.5)


@pytest.mark.parametrize("receiver", [0, 1, 2])
def test_fedavg_limit_unequal_sizes(receiver):
    sizes = np.array([10.0, 30.0, 60.0])
    p = sizes / sizes.sum()
    rng = np.random.default_rng(receiver)
    th = [rng.standard_normal(50) for _ in range(3)]
    w = donor_weights(receiver, q=[0.9, 0.1, 0.5], p=p, alpha=0.0, beta=1.0)
    out = aggregate(th, receiver, w, gamma=p[receiver])
    np.testing.assert_allclose(out, sum(pk * t for pk, t in zip(p, th)), atol=1e-8, rtol=0)


def test_missing_score_falls_back_to_size_anchor():
    p = np.array([0.2, 0.3, 0.5])
    w = donor_weights(0, q=[None, None, None], p=p, alpha=0.7, beta=1.0)
    np.testing.assert_allclose(w, [0.0, 0.375, 0.625])


def test_weights_sum_to_one_and_exclude_receiver():
    rng = np.random.default_rng(1)
    q = list(rng.random(6))
    q[2] = None
    w = donor_weights(3, q=q, p=np.full(6, 1 / 6), alpha=0.5, beta=1.0)
    assert w[3] == 0.0 and w.sum() == pytest.approx(1.0) and np.all(w >= 0)
