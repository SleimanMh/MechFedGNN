import numpy as np
import torch

from model import Standardiser, init_params, train_early_stopping

MC = {"hidden": [8], "lr": 1e-3, "batch": 16}


def test_robust_standardiser_uses_median_iqr_and_keeps_tails():
    x = np.concatenate([np.arange(1.0, 100.0), [10_000.0]])          # one huge outlier
    X = np.stack([x, np.ones_like(x)], 1)
    M = np.ones_like(X, dtype=np.int8)
    st = Standardiser(X, M, x)
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    assert st.mu[0] == med and st.sd[0] == q75 - q25                  # outlier does not set the scale
    z = st.inputs(X, M).numpy()[:, 0]
    assert z.max() > 100                                              # tail preserved, no clip
    assert st.sd[1] == 1.0                                            # constant: IQR 0, std 0 -> 1


def test_robust_standardiser_ignores_missing_entries():
    X = np.array([[1.0], [2.0], [3.0], [1000.0]])
    M = np.array([[1], [1], [1], [0]], dtype=np.int8)
    st = Standardiser(X, M, X[:, 0])
    assert st.mu[0] == 2.0
    assert st.inputs(X, M).numpy()[3, 0] == 0.0                        # missing -> 0


def test_iqr_zero_falls_back_to_std():
    x = np.array([0.0] * 8 + [5.0, 10.0])                              # >75% zeros: IQR 0
    st = Standardiser(x[:, None], np.ones((10, 1), dtype=np.int8), x)
    assert st.sd[0] == x.std()


def test_early_stopping_respects_min_steps_and_patience():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((64, 4)), dtype=torch.float32)
    y = torch.tensor(rng.standard_normal(64), dtype=torch.float32)     # pure noise: val never improves much
    theta = init_params(2, 0, (8,))
    es = {"eval_every": 5, "patience": 2, "min_steps": 100, "max_steps": 1000}
    _, best, halt = train_early_stopping(theta, 2, x, y, x[:16], y[:16] + 5.0, 0, MC, es)
    assert halt >= 100
    es_long = {**es, "patience": 50}
    _, _, halt_long = train_early_stopping(theta, 2, x, y, x[:16], y[:16] + 5.0, 0, MC, es_long)
    assert halt_long >= best + 50 * 5 or halt_long == 1000
