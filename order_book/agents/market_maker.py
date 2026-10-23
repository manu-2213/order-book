"""Avellaneda-Stoikov market maker with an optional detector-driven response."""

import math

from order_book.agents.base import Agent
from order_book.config import MarketMakerConfig
from order_book.detectors.base import Detector


class MarketMaker(Agent):
    """Two-sided post-only quoter in the spirit of Avellaneda-Stoikov.

    reservation = mid - inventory * gamma * var * horizon
    spread      = gamma * var * horizon + (2 / gamma) * ln(1 + gamma / kappa)

    ``var`` is the fast EWMA variance of mid changes from PublicState, floored
    at ``min_var``; the inventory skew and the half-spread are each capped at
    ``max_skew`` ticks so a volatility spike cannot make the quoter cross the
    book. A side is only cancelled and replaced when its target price moves at
    least one tick from the resting quote or its size has run below
    ``refill_frac`` of target, so the quoter keeps queue position. The side
    that would push inventory past ``max_inventory`` is pulled.

    With a ``detector``, its signal s in [-1, 1] skews the reservation
    ``detect_skew * s`` ticks, widens the pressured side by
    ``detect_widen * |s|`` and cuts its size by ``detect_size_cut * |s|``.
    The response mapping is fixed here; only the detector varies between
    adversary levels.

    Fragility: with r = sqrt(var_fast / var_slow), quote sizes on both sides
    are scaled by max(0, 1 - fragility * max(r - 1, 0)); when r exceeds
    ``pull_ratio`` both quotes are cancelled and stay away for ``cooldown``
    steps. Liquidity therefore thins exactly when a large order makes the
    price move fast.
    """

    def __init__(self, *args, cfg: MarketMakerConfig | None = None, detector: Detector | None = None):
        super().__init__(*args)
        self.cfg = cfg or MarketMakerConfig()
        self.detector = detector
        self.target = self.account.shares
        self.base_size = {"bid": self._draw_size(), "ask": self._draw_size()}
        self.quotes: dict[str, int | None] = {"bid": None, "ask": None}  # own order id per side
        self.pulled_until = 0

    def _draw_size(self) -> int:
        return int(self.rng.integers(self.cfg.size[0], self.cfg.size[1] + 1))

    def _requote(self, side: str, order, target_price: float, scale: float, allowed: bool) -> None:
        m = self.market
        target = max(1, round(self.base_size[side] * scale))
        if order is not None:
            if (allowed and abs(order["price"] - target_price) < 1.0
                    and self.cfg.refill_frac * target <= order["quantity"] <= target):
                return
            m.cancel(self.cid, order["order_id"])
        self.quotes[side] = None
        if allowed:
            self.base_size[side] = self._draw_size()
            report = m.limit(self.cid, side, round(target_price), max(1, round(self.base_size[side] * scale)))
            if report is not None and report["resting_quantity"]:
                self.quotes[side] = report["order_id"]

    def act(self) -> None:
        m, ps, book, cfg = self.market, self.market.public, self.market.book, self.cfg
        inventory = self.account.shares - self.target
        relative_vol = math.sqrt(ps.var_fast / ps.var_slow) if ps.var_slow > 0 else 1.0
        if relative_vol > cfg.pull_ratio:
            self.pulled_until = m.step + cfg.cooldown
        if m.step < self.pulled_until:
            for side in ("bid", "ask"):
                if self.quotes[side] is not None and book.get_order(self.quotes[side]) is not None:
                    m.cancel(self.cid, self.quotes[side])
                self.quotes[side] = None
            return
        fragile = max(0.0, 1.0 - cfg.fragility * max(relative_vol - 1.0, 0.0))
        var = max(ps.var_fast, cfg.min_var)
        risk = cfg.gamma * var * cfg.horizon
        reservation = ps.mid - max(-cfg.max_skew, min(cfg.max_skew, inventory * risk))
        half = min(max((risk + (2 / cfg.gamma) * math.log1p(cfg.gamma / cfg.kappa)) / 2, 1.0), cfg.max_skew)
        widen = {"bid": 0.0, "ask": 0.0}
        scale = {"bid": fragile, "ask": fragile}
        signal = self.detector.signal if self.detector is not None else 0.0
        if signal:
            pressured = "ask" if signal > 0 else "bid"  # buyers hit asks, sellers hit bids
            reservation += cfg.detect_skew * signal
            widen[pressured] = cfg.detect_widen * abs(signal)
            scale[pressured] *= max(1.0 - cfg.detect_size_cut * abs(signal), 0.0)
        bid = reservation - half - widen["bid"]
        ask = max(reservation + half + widen["ask"], bid + 2)
        own = {s: book.get_order(q) if q is not None else None for s, q in self.quotes.items()}
        # Post-only: never cross anyone else's resting order.
        for side in ("bid", "ask"):
            best = book.best_excluding(side, self.quotes[side])
            if best is not None:
                if side == "ask":
                    bid = min(bid, best - 1.0)
                else:
                    ask = max(ask, best + 1.0)
        self._requote("bid", own["bid"], bid, scale["bid"], inventory < cfg.max_inventory and scale["bid"] > 0)
        self._requote("ask", own["ask"], ask, scale["ask"], inventory > -cfg.max_inventory and scale["ask"] > 0)
