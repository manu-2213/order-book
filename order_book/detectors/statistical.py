"""Held-out statistical detectors (adversary level A2): different statistics from the rule detector,
never used in training.

    RunsDetector         runs of same-sign aggressive trades
    CusumDetector        two-sided CUSUM on the per-step normalised signed flow (online change-point)
    VolumeClockDetector  flow imbalance over the last ``volume`` lots instead of a time window
    RegularityDetector   metaorder reconstruction in miniature: a persistent same-sign run with
                         regular child sizes (in the style of Goliath & Gebbie, 2026)
"""

from order_book.detectors.base import Detector
from order_book.detectors.features import FeatureTracker
from order_book.public_state import PublicState


class _FeatureDetector(Detector):
    def __init__(self):
        self.tracker = FeatureTracker()
        self.signal = self.p_buy = self.p_sell = 0.0

    def _emit(self, signal: float) -> float:
        self.signal = max(-1.0, min(1.0, signal))
        self.p_buy, self.p_sell = max(self.signal, 0.0), max(-self.signal, 0.0)
        return self.signal


class RunsDetector(_FeatureDetector):
    """signal = sign(run) * clip((run_length - threshold) / scale, 0, 1)."""

    def __init__(self, threshold: int = 4, scale: int = 6):
        super().__init__()
        self.threshold, self.scale = threshold, scale

    def update(self, public: PublicState) -> float:
        self.tracker.update(public)
        t = self.tracker
        strength = (t.run_length - self.threshold) / self.scale
        return self._emit(t.run_sign * strength if strength > 0 else 0.0)


class CusumDetector(Detector):
    """Two-sided CUSUM with restart on the flow imbalance of consecutive ``block``-step blocks.

    Per block, x = (buy - sell) / (buy + sell) over the block's aggressive volume (0 without trades);
    S+ = max(0, S+ + x - k), S- = max(0, S- - x - k). When either statistic exceeds ``h`` the detector
    raises an alarm in that direction, restarts both statistics (the classical online change-point
    rule) and holds the alarm as a signal that decays linearly to zero over ``hold`` blocks; between
    alarms the signal is the partial evidence (S+ - S-) / h. With about 20 trades per block the block
    imbalance has a standard deviation near 0.25, so ``k`` = 0.1 is half the shift of a
    20%-participation metaorder and ``h`` = 1.5 fires after roughly 1.5 / (participation - k) blocks.
    """

    def __init__(self, k: float = 0.1, h: float = 1.5, block: int = 20, hold: int = 10):
        self.k, self.h, self.block, self.hold = k, h, block, hold
        self.s_plus = self.s_minus = 0.0
        self._buy = self._sell = self._n = 0
        self.alarm_sign, self.blocks_since_alarm = 0, hold
        self.signal = self.p_buy = self.p_sell = 0.0

    def update(self, public: PublicState) -> float:
        self._buy += sum(q for s, _, q in public.step_trades if s == "bid")
        self._sell += sum(q for s, _, q in public.step_trades if s == "ask")
        self._n += 1
        if self._n < self.block:
            return self.signal
        total = self._buy + self._sell
        x = (self._buy - self._sell) / total if total else 0.0
        self._buy = self._sell = self._n = 0
        self.s_plus = max(0.0, self.s_plus + x - self.k)
        self.s_minus = max(0.0, self.s_minus - x - self.k)
        self.blocks_since_alarm += 1
        if self.s_plus > self.h or self.s_minus > self.h:
            self.alarm_sign = 1 if self.s_plus > self.s_minus else -1
            self.blocks_since_alarm = 0
            self.s_plus = self.s_minus = 0.0
        held = self.alarm_sign * max(0.0, 1.0 - self.blocks_since_alarm / self.hold)
        partial = (self.s_plus - self.s_minus) / self.h
        self.signal = max(-1.0, min(1.0, held if abs(held) >= abs(partial) else partial))
        self.p_buy, self.p_sell = max(self.signal, 0.0), max(-self.signal, 0.0)
        return self.signal


class VolumeClockDetector(_FeatureDetector):
    """Imbalance over the last ``volume`` lots, thresholded like the flow detector."""

    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold

    def update(self, public: PublicState) -> float:
        f = self.tracker.update(public)
        imbalance = f[3]
        strength = (abs(imbalance) - self.threshold) / (1 - self.threshold)
        return self._emit((1 if imbalance > 0 else -1) * strength if strength > 0 else 0.0)


class RegularityDetector(_FeatureDetector):
    """A same-sign run of at least ``min_run`` trades with regular sizes: signal = sign * regularity *
    min(1, run / saturate_run)."""

    def __init__(self, min_run: int = 3, saturate_run: int = 6):
        super().__init__()
        self.min_run, self.saturate_run = min_run, saturate_run

    def update(self, public: PublicState) -> float:
        self.tracker.update(public)
        t = self.tracker
        if t.run_length < self.min_run:
            return self._emit(0.0)
        return self._emit(t.run_sign * t.regularity() * min(1.0, t.run_length / self.saturate_run))


def held_out_detectors() -> dict[str, Detector]:
    """The A2 set, by name."""
    return {"runs": RunsDetector(), "cusum": CusumDetector(), "volume_clock": VolumeClockDetector(),
            "regularity": RegularityDetector()}
