"""Held-out (A2) and learned (A3) detectors and their shared public features."""

import numpy as np
import pytest

from order_book.detectors import (
    FEATURE_NAMES, CusumDetector, FeatureTracker, LearnedDetector, LearnedDetectorParams, RegularityDetector,
    RunsDetector, VolumeClockDetector, held_out_detectors,
)
from order_book.detectors.learned import softmax, train
from order_book.scoring import roc_auc


class FakePublic:
    """Minimal PublicState stand-in: a list of (side, price, qty) trades per step."""

    def __init__(self, trades, mid=10_000.0, imbalance=0.0):
        self.step_trades, self.mid, self.imbalance = trades, mid, imbalance
        self.var_fast, self.spread, self.volume_slow = 0.05, 1, 5.0


def feed(detector, sides, qty=5, steps_each=1):
    out = []
    for side in sides:
        for _ in range(steps_each):
            out.append(detector.update(FakePublic([(side, 10_000, qty)], imbalance=1.0 if side == "bid" else -1.0)))
    return out


def test_feature_tracker_runs_regularity_and_shapes():
    tracker = FeatureTracker()
    for _ in range(5):
        f = tracker.update(FakePublic([("bid", 10_000, 10)]))
    assert f.shape == (len(FEATURE_NAMES),)
    assert tracker.run_sign == 1 and tracker.run_length == 5
    assert tracker.regularity() == pytest.approx(1.0)  # identical sizes
    assert f[3] == pytest.approx(1.0)                  # volume-clock imbalance all buys
    tracker.update(FakePublic([("ask", 10_000, 10)]))
    assert tracker.run_sign == -1 and tracker.run_length == 1 and tracker.regularity() == 0.0
    for _ in range(3):
        tracker.update(FakePublic([("ask", 10_000, 2), ("ask", 10_000, 30)]))
    assert 0.0 <= tracker.regularity() < 0.6  # very irregular sizes


@pytest.mark.parametrize("name", ["runs", "volume_clock", "regularity"])
def test_statistical_detectors_follow_one_sided_flow_and_flip(name):
    det = held_out_detectors()[name]
    sig = feed(det, ["bid"] * 15)
    assert sig[-1] > 0.3 and det.p_buy == sig[-1] and det.p_sell == 0.0
    sig = feed(det, ["ask"] * 40)
    assert sig[-1] < -0.3
    assert all(-1 <= s <= 1 for s in sig)


def test_cusum_alarms_restarts_and_decays():
    det = CusumDetector(k=0.1, h=1.5, block=20, hold=10)
    sig = feed(det, ["bid"] * 20 * 4)                 # four one-sided blocks: x = 1 each
    assert sig[-1] == 1.0                             # alarm raised and held
    assert det.s_plus == 0.0 and det.s_minus == 0.0 or det.blocks_since_alarm >= 0
    sig = []
    for _ in range(20 * 12):                           # quiet: alarm decays over ``hold`` blocks
        sig.append(det.update(FakePublic([])))
    assert sig[-1] == 0.0 and sig[0] > sig[-1]
    sig = feed(det, ["ask"] * 20 * 4)
    assert sig[-1] == -1.0


def test_learned_detector_trains_on_separable_data_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 3, n)
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    x[:, 0] += np.where(y == 1, 2.0, np.where(y == 2, -2.0, 0.0))   # imbalance separates the classes
    params = train(x[:2000], y[:2000], x[2000:], y[2000:], epochs=10, batch_size=128, verbose=False)
    p = softmax(((x[2000:] - params.mean) / params.std) @ params.weights + params.bias)
    assert roc_auc(p[:, 1], y[2000:] == 1) > 0.9 and roc_auc(p[:, 2], y[2000:] == 2) > 0.9
    params.save(tmp_path / "w.npz")
    loaded = LearnedDetectorParams.load(str(tmp_path / "w.npz"))
    assert np.allclose(loaded.weights, params.weights) and loaded.feature_names == FEATURE_NAMES
    det = LearnedDetector(str(tmp_path / "w.npz"))
    sig = feed(det, ["bid"] * 10)
    assert det.p_buy > det.p_sell and sig[-1] == pytest.approx(det.p_buy - det.p_sell)
    sig = feed(det, ["ask"] * 30)
    assert det.p_sell > det.p_buy


def test_observers_attach_to_the_simulator_and_update_every_step():
    from order_book.experiments import make_observers
    from order_book.simulator import Simulator
    observers = make_observers(("flow_lenient", "runs", "cusum", "volume_clock", "regularity", "learned"), None)
    assert set(observers) == {"flow_lenient", "runs", "cusum", "volume_clock", "regularity"}  # A3 skipped: no weights
    sim = Simulator(0, analysis=False)
    for name, det in observers.items():
        sim.detectors[f"obs:{name}"] = det
    sim.run(300)
    assert all(abs(det.signal) <= 1 for det in observers.values())
    assert observers["runs"].tracker.run_length > 0
