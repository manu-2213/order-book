"""Learned detector (adversary level A3): a multinomial logistic classifier on public features.

Trained offline on background-market runs, with labels from the institutional
metaorders only (plan section 6): class 0 = no metaorder active, 1 = a buy
metaorder active, 2 = a sell metaorder active (net side when several
overlap). The detector is never trained against an executor, so its score on
the executor measures how much the executor looks like an institution.

Training is plain NumPy (Adam on the cross-entropy with inverse-frequency
class weights) and prints one line per epoch: loss, one-vs-rest ROC AUC for
the buy and sell classes on held-out seeds.
"""

import json
import time
from dataclasses import dataclass, replace

import numpy as np

from order_book.config import SimConfig
from order_book.detectors.base import Detector
from order_book.detectors.features import FEATURE_NAMES, FeatureTracker
from order_book.public_state import PublicState


@dataclass
class LearnedDetectorParams:
    weights: np.ndarray            # (n_features, 3)
    bias: np.ndarray               # (3,)
    mean: np.ndarray               # feature standardisation
    std: np.ndarray
    feature_names: tuple = FEATURE_NAMES

    def save(self, path) -> None:
        np.savez(path, weights=self.weights, bias=self.bias, mean=self.mean, std=self.std,
                 feature_names=np.array(self.feature_names))

    @staticmethod
    def load(path) -> "LearnedDetectorParams":
        d = np.load(path, allow_pickle=False)
        return LearnedDetectorParams(d["weights"], d["bias"], d["mean"], d["std"], tuple(d["feature_names"].tolist()))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LearnedDetector(Detector):
    """signal = p(buy active) - p(sell active) from the fitted classifier."""

    def __init__(self, params: LearnedDetectorParams | str):
        self.params = LearnedDetectorParams.load(params) if isinstance(params, str) else params
        self.tracker = FeatureTracker()
        self.signal = self.p_buy = self.p_sell = 0.0

    def update(self, public: PublicState) -> float:
        f = self.tracker.update(public)
        z = ((f - self.params.mean) / self.params.std) @ self.params.weights + self.params.bias
        p = softmax(z)
        self.p_buy, self.p_sell = float(p[1]), float(p[2])
        self.signal = self.p_buy - self.p_sell
        return self.signal


# ----- dataset ---------------------------------------------------------------------------- #

def collect_dataset(seeds, steps: int, config: SimConfig | None = None, burn_in: int = 2_000,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-step features and labels (0 none, 1 buy, 2 sell) from background-market runs."""
    from order_book.simulator import Simulator  # local import: keeps detectors importable without the simulator

    xs, ys = [], []
    for seed in seeds:
        t0 = time.perf_counter()
        sim = Simulator(int(seed), config, analysis=True)
        tracker = FeatureTracker()
        sim.run(burn_in)
        x = np.empty((steps, len(FEATURE_NAMES)))
        active = {}
        for t in range(steps):
            sim.step()
            x[t] = tracker.update(sim.market.public)
        for r in sim.metaorders.records:  # labels from the ground truth, after the fact
            end = r.end if r.end is not None else sim.market.step
            for u in range(max(r.start, burn_in), min(end, burn_in + steps)):
                active[u - burn_in] = active.get(u - burn_in, 0) + (1 if r.side == "bid" else -1)
        y = np.zeros(steps, dtype=int)
        for t, net in active.items():
            y[t] = 1 if net > 0 else (2 if net < 0 else 0)
        xs.append(x)
        ys.append(y)
        if verbose:
            share = (y > 0).mean()
            print(f"[dataset] seed {seed}: {steps} steps, {share * 100:.1f}% of steps with a metaorder active "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


# ----- training --------------------------------------------------------------------------- #

def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney), ties handled by average ranks."""
    from order_book.scoring import roc_auc as _auc
    return _auc(scores, labels)


def train(x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
          epochs: int = 30, lr: float = 0.02, batch_size: int = 4096, l2: float = 1e-4, seed: int = 0,
          verbose: bool = True) -> LearnedDetectorParams:
    rng = np.random.default_rng(seed)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-8
    xn = (x - mean) / std
    n, d = xn.shape
    counts = np.bincount(y, minlength=3).astype(float)
    class_weight = counts.sum() / (3 * np.maximum(counts, 1))
    w = np.zeros((d, 3))
    b = np.zeros(3)
    m_w, v_w, m_b, v_b = np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    onehot = np.eye(3)[y]
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb, ob = xn[idx], y[idx], onehot[idx]
            p = softmax(xb @ w + b)
            wt = class_weight[yb]
            total += float(-(wt * np.log(p[np.arange(len(idx)), yb] + 1e-12)).sum())
            g = (p - ob) * wt[:, None] / len(idx)
            gw = xb.T @ g + l2 * w
            gb = g.sum(axis=0)
            step += 1
            for param, grad, m, v in ((w, gw, m_w, v_w), (b, gb, m_b, v_b)):
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * grad * grad
                param -= lr * (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + eps)
        if verbose:
            msg = f"[train] epoch {epoch:3d}/{epochs}  loss {total / n:.4f}"
            if x_val is not None:
                p = softmax(((x_val - mean) / std) @ w + b)
                msg += (f"  val AUC buy {roc_auc(p[:, 1], (y_val == 1).astype(int)):.3f}"
                        f"  sell {roc_auc(p[:, 2], (y_val == 2).astype(int)):.3f}"
                        f"  signal {roc_auc(np.abs(p[:, 1] - p[:, 2]), (y_val > 0).astype(int)):.3f}")
            print(msg + f"  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return LearnedDetectorParams(w, b, mean, std)
