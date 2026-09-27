import numpy as np
import pandas as pd

from data.clients import client_profiles, group_profiles
from data.inject import MECHANISMS, build_roles, expected_feature_rates, inject, panel_probs_for_rate
from data.validate_injector import precondition
from signatures import phi
from tests.fixtures import synthetic


def _within_cross(M, roles):
    C = phi(M)
    pid = {f: k for k, P in enumerate(roles["panels"]) for f in P}
    mk = roles["maskable"]
    w, c = [], []
    for a in range(len(mk)):
        for b in range(a + 1, len(mk)):
            (w if pid[mk[a]] == pid[mk[b]] else c).append(C[mk[a], mk[b]])
    return np.mean(w), np.mean(c)


def _cross(X, y, roles, mech, seeds, **kw):
    p = panel_probs_for_rate(len(roles["panels"]), 0.3, 0.05)
    return np.mean([_within_cross(inject(X, y, roles, p, mech, np.random.default_rng(s),
                                         driver_seed=s, **kw)[0], roles)[1] for s in seeds])


def test_panel_jitter_zero_members_identical():
    X, y, roles = synthetic()
    M, _ = inject(X, y, roles, [0.7, 0.6], "mcar", np.random.default_rng(0), jitter=0.0)
    for P in roles["panels"]:
        for f in P[1:]:
            assert np.array_equal(M[:, P[0]], M[:, f])
    within, _ = _within_cross(M, roles)
    assert within == 1.0
    assert np.all(M[:, roles["always_observed"]] == 1)


def test_panel_jitter_keeps_within_phi_high_but_below_one():
    X, y, roles = synthetic()
    M, _ = inject(X, y, roles, [0.7, 0.6], "mcar", np.random.default_rng(0), jitter=0.05)
    within, _ = _within_cross(M, roles)
    assert 0.6 < within < 1.0


def test_null_mcar_cross_phi_within_3_se():
    X, y, roles = synthetic()
    p = panel_probs_for_rate(2, 0.3, 0.05)
    M, _ = inject(X, y, roles, p, "mcar", np.random.default_rng(11), driver_overlap=0.0)
    _, cross = _within_cross(M, roles)
    p_obs = M[:, roles["maskable"]].mean()
    se = np.sqrt((1 - p_obs**2) / (len(M) * p_obs**2))
    assert abs(cross) <= 3 * se


def test_dose_response_driver_overlap():
    X, y, roles = synthetic()
    vals = [_cross(X, y, roles, "mar", range(3), driver_overlap=o) for o in [0, .25, .5, .75, 1]]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_dose_response_class_spread():
    X, y, roles = synthetic()
    vals = [_cross(X, y, roles, "cd_mnar", range(3), class_spread=s) for s in [0, .25, .5, .75, 1]]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_driver_overlap_continuous_monotonicity():
    """Fine grid, common random numbers: cross-panel phi rises at every 0.1 step."""
    X, y, roles = synthetic(n=6000)
    grid = np.round(np.linspace(0, 1, 11), 2)
    vals = [_cross(X, y, roles, "mar", range(3), driver_overlap=o) for o in grid]
    assert all(b > a for a, b in zip(vals, vals[1:])), dict(zip(grid, np.round(vals, 4)))
    assert abs(vals[0]) < 0.03 and vals[-1] > 0.2


def test_rate_calibration_all_mechanisms():
    X, y, roles = synthetic()
    p = panel_probs_for_rate(2, 0.3, 0.05)
    kw = {"cell": {}, "mcar": {}, "mar": {"driver_overlap": 0.5}, "fd_mnar": {"direction": "top"},
          "cd_mnar": {"class_spread": 0.5}}
    rates = [1 - inject(X, y, roles, p, m, np.random.default_rng(0), **kw[m])[0][:, roles["maskable"]].mean()
             for m in MECHANISMS]
    assert max(rates) - min(rates) <= 0.02, rates


def test_panels_identical_across_clients_ordering_differs():
    X, y, roles = synthetic(n=6000)
    prof, groups = client_profiles(len(roles["panels"]), np.random.default_rng(0))
    parts = np.array_split(np.arange(len(X)), len(prof))
    ordered_rate = []
    for k, rows in enumerate(parts):
        M, ordered = inject(X[rows], y[rows], roles, prof[k], "mcar", np.random.default_rng(k), jitter=0.0)
        for P in roles["panels"]:                 # same panel membership in every client
            for f in P[1:]:
                assert np.array_equal(M[:, P[0]], M[:, f])
        ordered_rate.append(ordered.mean(0))
    ordered_rate = np.array(ordered_rate)
    assert len({tuple(np.round(r, 6)) for r in prof}) == len(prof)      # every client differs
    a, b = ordered_rate[groups == 0], ordered_rate[groups == 1]
    assert a[:, 0].max() < b[:, 0].min() and b[:, 1].max() < a[:, 1].min()


def test_group_profiles_halves_and_middle_control():
    np.testing.assert_allclose(group_profiles(2), [[0.2, 0.9], [0.9, 0.2]])
    np.testing.assert_allclose(group_profiles(3), [[0.2, 0.9, 0.9], [0.9, 0.9, 0.2]])
    np.testing.assert_allclose(group_profiles(4), [[0.2, 0.2, 0.9, 0.9], [0.9, 0.9, 0.2, 0.2]])


def test_constant_columns_never_maskable_or_characteristic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((500, 6)), columns=[f"f{i}" for i in range(6)])
    df["f2"] = 3.0
    df["target"] = df[["f0", "f1"]].sum(1)
    roles = build_roles(df)
    assert roles["constant"] == [2]
    assert 2 not in roles["maskable"] and 2 not in roles["always_observed"]
    assert "f2" not in roles["bin_edges"]


def test_precondition_rejects_collinear_all_or_nothing_single_panel_or_small():
    good = {"panel_corr_median": 0.3, "panels": [[0], [1]]}
    assert precondition(good, {"mcar": "ok"}, [150] * 6) == (True, [])
    ok, why = precondition({**good, "panel_corr_median": 0.95}, {"mcar": "ok"})
    assert not ok and "0.950 > 0.9" in why[0]
    ok, why = precondition(good, {"fd_mnar top": "ALL-OR-NOTHING"})
    assert not ok and "fd_mnar top" in why[0]
    ok, why = precondition({**good, "panels": [[0, 1]]}, {})
    assert not ok and "no latent group structure" in why[0]
    ok, why = precondition(good, {}, [150, 150, 99])
    assert not ok and "99 < 100" in why[0]


def test_cell_masking_is_the_independence_control():
    """(a) in §3.6: rate-matched to the panel conditions in expectation, no excess
    association anywhere - but joint absence is NOT zero: it sits at r_f * r_g."""
    X, y, roles = synthetic(n=20000)
    p = panel_probs_for_rate(2, 0.3, 0.05)
    exp = expected_feature_rates(roles, p, 0.05)
    M, ordered = inject(X, y, roles, p, "cell", np.random.default_rng(0), jitter=0.05)
    assert ordered is None
    real = 1 - M.mean(0)
    np.testing.assert_allclose(real[roles["maskable"]], exp[roles["maskable"]], atol=0.01)
    within, cross = _within_cross(M, roles)
    assert abs(within) < 0.03 and abs(cross) < 0.03
    A = 1 - M[:, 0].astype(float), 1 - M[:, 1].astype(float)
    assert abs((A[0] * A[1]).mean() - real[0] * real[1]) < 0.01    # ~0.09, not 0
    Mp, _ = inject(X, y, roles, p, "mcar", np.random.default_rng(0), jitter=0.05)
    assert _within_cross(Mp, roles)[0] > 0.6                       # (b) keeps within-panel dependence
