"""Rule-based detector on the imbalance of signed aggressive flow."""

import math

from order_book.config import FlowDetectorConfig
from order_book.detectors.base import Detector
from order_book.public_state import PublicState


class FlowDetector(Detector):
    """EWMA of the rolling flow imbalance, thresholded and rescaled.

    ``memory`` follows ``PublicState.imbalance`` with span ``cfg.memory``;
    the signal is 0 while |memory| < threshold and grows linearly to +/-1 as
    |memory| reaches 1. ``p_buy`` / ``p_sell`` are the positive / negative
    parts of the signal. With ``include_hidden`` it reads ``imbalance_all``,
    so it sees traders hidden by ``Market.hide``; the channel decomposition
    uses this to blind makers and the anticipator independently.
    """

    def __init__(self, cfg: FlowDetectorConfig | None = None, include_hidden: bool = False):
        cfg = cfg or FlowDetectorConfig()
        self.memory = self.signal = self.p_buy = self.p_sell = 0.0
        self._alpha, self._threshold = 2 / (cfg.memory + 1), cfg.threshold
        self.include_hidden = include_hidden

    def update(self, public: PublicState) -> float:
        imbalance = public.imbalance_all if self.include_hidden else public.imbalance
        self.memory += self._alpha * (imbalance - self.memory)
        strength = (abs(self.memory) - self._threshold) / (1 - self._threshold)
        self.signal = math.copysign(min(strength, 1.0), self.memory) if strength > 0 else 0.0
        self.p_buy, self.p_sell = max(self.signal, 0.0), max(-self.signal, 0.0)
        return self.signal
