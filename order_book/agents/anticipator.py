"""Order anticipator: a trader that acts on a detector's signal."""

from order_book.agents.base import Agent
from order_book.config import OrderAnticipatorConfig
from order_book.detectors.base import Detector


class OrderAnticipator(Agent):
    """Trades in the direction of a detected metaorder and unwinds when the signal fades.

    With |signal| >= ``threshold`` it sends market orders of up to ``chunk``
    lots at ``rate`` x (|signal| - threshold) / (1 - threshold) lots per step
    in the signal's direction, holding within ``max_position``. With
    |signal| < ``fade`` it unwinds ``chunk`` lots per step. In between it
    holds. With ``contrarian_first`` it trades *against* the flow (supplying
    liquidity at the start, as van Kervel & Menkveld report) until the
    signal has been confident for ``switch_after`` steps, then with it.
    A false alarm therefore costs it the round trip.
    """

    def __init__(self, *args, cfg: OrderAnticipatorConfig | None = None, detector: Detector):
        super().__init__(*args)
        self.cfg = cfg or OrderAnticipatorConfig()
        self.detector = detector
        self.target = self.account.shares
        self.confident_steps = 0

    def act(self) -> None:
        m, cfg, signal = self.market, self.cfg, self.detector.signal
        position = self.account.shares - self.target
        strength = abs(signal)
        if strength >= cfg.threshold:
            self.confident_steps += 1
            direction = 1 if signal > 0 else -1
            if cfg.contrarian_first and self.confident_steps <= cfg.switch_after:
                direction = -direction
            quantity = min(cfg.chunk, round(cfg.rate * (strength - cfg.threshold) / (1 - cfg.threshold)))
            room = cfg.max_position - position if direction > 0 else cfg.max_position + position
            quantity = min(quantity, room)
            if quantity >= 1:
                m.market(self.cid, "bid" if direction > 0 else "ask", quantity)
        elif strength < cfg.fade:
            self.confident_steps = 0
            if position:
                m.market(self.cid, "ask" if position > 0 else "bid", min(cfg.chunk, abs(position)))
