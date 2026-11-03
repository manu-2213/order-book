"""Avellaneda-Stoikov market maker with an optional detector-driven response."""

import math

from order_book.agents.base import Agent
from order_book.config import MarketMakerConfig
from order_book.detectors.base import Detector
from order_book.rng import CONSTRUCTION, MM_SIZE_ASK, MM_SIZE_BID, keyed

_SIZE_PURPOSE = {"bid": MM_SIZE_BID, "ask": MM_SIZE_ASK}


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

    Inventory freeze: ``skew_account``, when set, is the account whose holding
    drives the reservation skew. The channel decomposition points it at the
    same maker's account in Arm 0, so the quotes stop leaning on the inventory
    the executor pushed onto this maker. The ``max_inventory`` pull is a risk
    limit, not part of the skew, and always uses the maker's real holding.

    ``public_view``, when set, is the ``PublicState`` the quotes are centred on
    and scaled by; the decomposition's control points it at Arm 0, so the maker
    keeps quoting in the true book but does not follow the executor's price move.
    """

    skew_account = None
    public_view = None

    def __init__(self, *args, cfg: MarketMakerConfig | None = None, detector: Detector | None = None):
        super().__init__(*args)
        self.cfg = cfg or MarketMakerConfig()
        self.detector = detector
        self.target = self.account.shares
        self.base_size = {"bid": self._draw_size("bid", CONSTRUCTION), "ask": self._draw_size("ask", CONSTRUCTION)}
        self.quotes: dict[str, int | None] = {"bid": None, "ask": None}  # own order id per side
        self.pulled_until = 0

    def _draw_size(self, side: str, step: int | None = None) -> int:
        step = self.market.step if step is None else step
        return int(keyed(self.rng, step, _SIZE_PURPOSE[side]).integers(self.cfg.size[0], self.cfg.size[1] + 1))

    def _skew(self, inventory: int, risk: float) -> float:
        return max(-self.cfg.max_skew, min(self.cfg.max_skew, inventory * risk))

    def _pressure(self, side: str, signal: float) -> tuple[float, float]:
        """(extra half-spread, size multiplier) a detector signal puts on one side of the quotes."""
        pressured = "ask" if signal > 0 else "bid"  # buyers hit asks, sellers hit bids
        if not signal or side != pressured:
            return 0.0, 1.0
        return self.cfg.detect_widen * abs(signal), max(1.0 - self.cfg.detect_size_cut * abs(signal), 0.0)

    def shade(self, side: str, price: int, quantity: int, inventory: int, signal: float) -> tuple[int, int] | None:
        """This maker's just-sent quote, had it held ``inventory`` and read ``signal`` instead of its own.

        Used by the ghost arm, where the maker's orders are replayed from Arm 0
        and only the terms that a switched-on channel changes may differ. The
        result equals the input when the terms equal the maker's own, so
        switching a channel on with no divergence changes nothing. ``None``
        means the size cut removes the quote.
        """
        cfg, public = self.cfg, self.market.public
        own_signal = self.detector.signal if self.detector is not None else 0.0
        risk = cfg.gamma * max(public.var_fast, cfg.min_var) * cfg.horizon
        shift = (self._skew(self.account.shares - self.target, risk) - self._skew(inventory, risk)
                 + cfg.detect_skew * (signal - own_signal))
        own_widen, own_scale = self._pressure(side, own_signal)
        widen, scale = self._pressure(side, signal)
        outward = widen - own_widen
        price = round(price + shift + (outward if side == "ask" else -outward))
        if scale == 0.0:
            return None
        return price, (quantity if scale == own_scale else max(1, round(quantity * scale / own_scale)))

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
            self.base_size[side] = self._draw_size(side)
            report = m.limit(self.cid, side, round(target_price), max(1, round(self.base_size[side] * scale)))
            if report is not None and report["resting_quantity"]:
                self.quotes[side] = report["order_id"]

    def act(self) -> None:
        m, ps, book, cfg = self.market, self.public_view or self.market.public, self.market.book, self.cfg
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
        skew_account = self.skew_account or self.account
        reservation = ps.mid - self._skew(skew_account.shares - self.target, risk)
        half = min(max((risk + (2 / cfg.gamma) * math.log1p(cfg.gamma / cfg.kappa)) / 2, 1.0), cfg.max_skew)
        widen = {"bid": 0.0, "ask": 0.0}
        scale = {"bid": fragile, "ask": fragile}
        signal = self.detector.signal if self.detector is not None else 0.0
        reservation += cfg.detect_skew * signal
        for side in ("bid", "ask"):
            widen[side], cut = self._pressure(side, signal)
            scale[side] *= cut
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
        # after a jump the new bid can reach the old ask (or the new ask the old bid): move that quote first, or the
        # book's self-trade prevention cancels the new one and the maker is one-sided for a step
        order = ("ask", "bid") if own["ask"] is not None and bid >= own["ask"]["price"] else ("bid", "ask")
        for side in order:
            allowed = (inventory < cfg.max_inventory) if side == "bid" else (inventory > -cfg.max_inventory)
            self._requote(side, own[side], bid if side == "bid" else ask, scale[side], allowed and scale[side] > 0)
