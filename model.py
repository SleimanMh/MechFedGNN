"""Mask-aware MLP, standardisation, training and evaluation (CLAUDE.md §1, §8).

Input is concat([x_zero_filled, M]) where x is standardised with the owning
client's training-fold statistics over OBSERVED entries; missing entries are 0
(= the training mean after standardisation). Loss is L_pred (MSE) only.
Parameters cross the model boundary as {name: float64 ndarray} so the kernel
can mix them with plain arithmetic.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

class MaskMLP(nn.Module):
    def __init__(self, d, hidden=(64, 32)):
        super().__init__()
        layers, width = [], 2 * d
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xin):
        return self.net(xin).squeeze(-1)


def init_params(d, seed, hidden=(64, 32)):
    """The shared theta_0 of one run."""
    torch.manual_seed(seed)
    return get_params(MaskMLP(d, hidden))


def get_params(model):
    return {k: v.detach().double().numpy().copy() for k, v in model.state_dict().items()}


def build(params, d, hidden=(64, 32)):
    model = MaskMLP(d, hidden)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in params.items()})
    return model


def _robust_centre_scale(x):
    """Median and IQR of the observed values of one feature. IQR 0 (a value
    covering > half the rows) falls back to the std; std 0 falls back to 1."""
    if len(x) == 0:
        return 0.0, 1.0
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    scale = q75 - q25
    if scale <= 0:
        scale = x.std()
    return float(med), float(scale) if scale > 0 else 1.0


class Standardiser:
    """Training-fold statistics of one client. Features: median / IQR over
    OBSERVED entries, no clipping, so heavy tails are preserved but a single
    extreme value cannot set the scale. Target: mean / std (the loss is MSE)."""

    def __init__(self, X, M, y):
        obs = M.astype(bool)
        stats = [_robust_centre_scale(X[obs[:, f], f]) for f in range(X.shape[1])]
        self.mu = np.array([m for m, _ in stats])
        self.sd = np.array([s for _, s in stats])
        self.y_mu, self.y_sd = float(y.mean()), float(y.std()) or 1.0
        self.y_median = float(np.median(y))

    def inputs(self, X, M):
        x = np.where(M.astype(bool), (X - self.mu) / self.sd, 0.0)
        return torch.tensor(np.concatenate([x, M], 1), dtype=torch.float32)

    def target(self, y):
        return torch.tensor((y - self.y_mu) / self.y_sd, dtype=torch.float32)

    def untarget(self, z):
        return z * self.y_sd + self.y_mu


def train(params, d, xin, yt, steps, seed, cfg, checkpoints=(), on_checkpoint=None):
    """Adam on MSE from `params` (fresh optimiser). Minibatches drawn from a
    generator seeded by `seed`, so the trajectory is deterministic and a
    checkpoint at step s equals a separate run of s steps."""
    model = build(params, d, cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gen = torch.Generator().manual_seed(int(seed))
    n, bs = len(yt), min(cfg["batch"], len(yt))
    marks = set(checkpoints)
    if 0 in marks and on_checkpoint:
        on_checkpoint(0, get_params(model))
    for step in range(1, steps + 1):
        idx = torch.randperm(n, generator=gen)[:bs]
        opt.zero_grad()
        loss = ((model(xin[idx]) - yt[idx]) ** 2).mean()
        loss.backward()
        opt.step()
        if step in marks and on_checkpoint:
            on_checkpoint(step, get_params(model))
    return get_params(model)


def train_early_stopping(params, d, xin, yt, xval, yval, seed, cfg, es):
    """Adam on MSE, validation MSE checked every `es['eval_every']` steps. A stop
    is permitted only from step `es['min_steps']` on, after `es['patience']`
    checks without improvement; `es['max_steps']` caps the run. Returns the best
    parameters, the step they were reached at, and the step training halted."""
    model = build(params, d, cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gen = torch.Generator().manual_seed(int(seed))
    n, bs = len(yt), min(cfg["batch"], len(yt))

    def val_loss():
        with torch.no_grad():
            return float(((model(xval) - yval) ** 2).mean())

    best, best_step, best_params, waited = val_loss(), 0, get_params(model), 0
    step = 0
    for step in range(1, es["max_steps"] + 1):
        idx = torch.randperm(n, generator=gen)[:bs]
        opt.zero_grad()
        ((model(xin[idx]) - yt[idx]) ** 2).mean().backward()
        opt.step()
        if step % es["eval_every"] == 0:
            v = val_loss()
            if v < best:
                best, best_step, best_params, waited = v, step, get_params(model), 0
            else:
                waited += 1
                if waited >= es["patience"] and step >= es["min_steps"]:
                    break
    return best_params, best_step, step


def predict(params, d, xin, st, hidden=(64, 32)):
    with torch.no_grad():
        z = build(params, d, hidden)(xin).numpy().astype(float)
    return st.untarget(z)


def metrics(y, pred, train_median):
    """RMSE, MAE, and AUC of the regression prediction against y > train-fold median."""
    err = pred - y
    pos = (y > train_median).astype(int)
    auc = roc_auc_score(pos, pred) if 0 < pos.sum() < len(pos) else float("nan")
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
            "auc": float(auc), "pos_rate": float(pos.mean())}
