import numpy as np
import pytest

from scores import combined_q, missingness_similarity, rate_similarity, w_c, w_h
from signatures import signature
from tests.fixtures import mask_a, mask_b, mask_c


@pytest.fixture
def sigs():
    return {k: signature(f()) for k, f in [("A", mask_a), ("B", mask_b), ("C", mask_c)]}


def test_fixture_joint_observation(sigs):
    assert sigs["B"]["J"][0, 1] == pytest.approx(0.9)
    assert sigs["C"]["J"][0, 1] == pytest.approx(0.3)


def test_w_h_one_positive_pair_equals_sender_J(sigs):
    assert w_h(sigs["A"]["H"], sigs["B"]["J"]) == (pytest.approx(0.9), True)
    assert w_h(sigs["A"]["H"], sigs["C"]["J"]) == (pytest.approx(0.3), True)


def test_w_c_one_positive_pair_equals_sender_J(sigs):
    assert w_c(sigs["A"]["C"], sigs["B"]["J"]) == (pytest.approx(0.9), True)
    assert w_c(sigs["A"]["C"], sigs["C"]["J"]) == (pytest.approx(0.3), True)


def test_w_undefined_when_receiver_has_no_joint_gaps(sigs):
    complete = signature(np.ones((50, 3), dtype=np.int8))
    assert w_h(complete["H"], sigs["B"]["J"]) == (None, False)
    assert w_c(complete["C"], sigs["B"]["J"]) == (None, False)


def test_w_is_directed(sigs):
    ac, _ = w_h(sigs["A"]["H"], sigs["C"]["J"])
    ca, _ = w_h(sigs["C"]["H"], sigs["A"]["J"])
    assert ac != pytest.approx(ca)


def test_symmetric_similarities(sigs):
    assert missingness_similarity(sigs["A"]["C"], sigs["A"]["C"]) == (pytest.approx(1.0), True)
    assert rate_similarity(sigs["A"]["r"], sigs["A"]["r"]) == (pytest.approx(1.0), True)
    s_ac, _ = missingness_similarity(sigs["A"]["C"], sigs["C"]["C"])
    s_ca, _ = missingness_similarity(sigs["C"]["C"], sigs["A"]["C"])
    assert s_ac == pytest.approx(s_ca)
    assert missingness_similarity(np.zeros((3, 3)), sigs["A"]["C"]) == (None, False)


def test_all_scores_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(30):
        si = signature((rng.random((200, 5)) < 0.6).astype(np.int8))
        sj = signature((rng.random((200, 5)) < 0.6).astype(np.int8))
        for v, ok in [w_h(si["H"], sj["J"]), w_c(si["C"], sj["J"]),
                      missingness_similarity(si["C"], sj["C"]), rate_similarity(si["r"], sj["r"])]:
            assert not ok or 0.0 <= v <= 1.0


# ---------------------------------------------------------------- fallback hierarchy

def test_fallback_both_available_uses_formula():
    assert combined_q(0.8, 0.4, lam_pop=0.25) == (pytest.approx(0.7), "formula")


def test_fallback_w_undefined_uses_S():
    assert combined_q(None, 0.4, lam_pop=0.25) == (pytest.approx(0.4), "S")


def test_fallback_S_none_uses_W():
    assert combined_q(0.8, None, lam_pop=0.25) == (pytest.approx(0.8), "W")


def test_fallback_neither_is_none():
    assert combined_q(None, None, lam_pop=0.25) == (None, "none")
