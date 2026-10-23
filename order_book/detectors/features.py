"""Public features for statistical and learned detectors, computed step by step from ``PublicState``.

Everything here is derived from the anonymous per-step trade list and the
public aggregates; no identities. The same tracker feeds the A2 detectors
and the A3 classifier, so they differ only in how they turn features into a
signal.
"""

import math
from collections import deque

import numpy as np

from order_book.public_state import PublicState

FEATURE_NAMES = (
    "imbalance_20", "imbalance_50", "imbalance_200", "imbalance_volume_clock", "signed_run",
    "regularity", "trade_rate_ratio", "normalised_return_50", "spread", "count_imbalance_20",
)


class FeatureTracker:
    """Rolling public statistics; ``update(public)`` returns the current feature vector."""

    def __init__(self, volume_clock: int = 100, long_window: int = 200, return_window: int = 50):
        self.volume_clock, self.long_window, self.return_window = volume_clock, long_window, return_window
        self._flow = deque(maxlen=long_window)        # (buy_volume, sell_volume, trade_count, buy_count) per step
        self._clock: deque = deque()                  # (side, qty) most recent aggressive trades, <= volume_clock lots
        self._clock_volume = 0
        self._mids = deque(maxlen=return_window + 1)
        self.run_sign, self.run_length = 0, 0
        self._run_sizes: deque = deque(maxlen=10)
        self.features = np.zeros(len(FEATURE_NAMES))

    def update(self, public: PublicState) -> np.ndarray:
        trades = public.step_trades
        buy = sum(q for s, _, q in trades if s == "bid")
        sell = sum(q for s, _, q in trades if s == "ask")
        self._flow.append((buy, sell, len(trades), sum(1 for s, _, _ in trades if s == "bid")))
        for side, _, qty in trades:
            sign = 1 if side == "bid" else -1
            if sign == self.run_sign:
                self.run_length += 1
            else:
                self.run_sign, self.run_length = sign, 1
                self._run_sizes.clear()
            self._run_sizes.append(qty)
            self._clock.append((sign, qty))
            self._clock_volume += qty
            while self._clock_volume - self._clock[0][1] >= self.volume_clock:
                self._clock_volume -= self._clock.popleft()[1]
        self._mids.append(public.mid)
        f = self.features
        f[0] = public.imbalance
        f[1] = self._imbalance(50)
        f[2] = self._imbalance(self.long_window)
        signed = sum(s * q for s, q in self._clock)
        f[3] = signed / self._clock_volume if self._clock_volume else 0.0
        f[4] = self.run_sign * min(self.run_length, 20) / 10
        f[5] = self.regularity()
        recent = sum(c for _, _, c, _ in list(self._flow)[-20:])
        long = sum(c for _, _, c, _ in self._flow)
        f[6] = (recent / 20) / (long / len(self._flow)) if long else 1.0
        if len(self._mids) > 1:
            scale = math.sqrt(max(public.var_fast, 1e-6) * (len(self._mids) - 1))
            f[7] = (self._mids[-1] - self._mids[0]) / scale
        else:
            f[7] = 0.0
        f[8] = public.spread
        rows = list(self._flow)[-20:]
        count = sum(c for _, _, c, _ in rows)
        f[9] = (2 * sum(b for _, _, _, b in rows) - count) / count if count else 0.0
        return f

    def _imbalance(self, window: int) -> float:
        rows = list(self._flow)[-window:]
        buy, sell = sum(b for b, _, _, _ in rows), sum(s for _, s, _, _ in rows)
        return (buy - sell) / (buy + sell) if buy + sell else 0.0

    def regularity(self) -> float:
        """1 - coefficient of variation of the sizes in the current same-sign run (0 with < 3 trades)."""
        if len(self._run_sizes) < 3:
            return 0.0
        sizes = np.fromiter(self._run_sizes, float)
        mean = sizes.mean()
        return float(max(0.0, 1.0 - sizes.std() / mean)) if mean > 0 else 0.0
