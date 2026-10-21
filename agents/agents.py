"""Agents for an agent-based limit order book simulation.

Accounting lives here, never in the book: every order goes through ``Market``,
which settles both sides of each trade. Prices are integer ticks, sizes lots.
Nobody may short or borrow: resting bids reserve cash, resting asks reserve
shares, so cash and shares are conserved across the whole population.

Agents only use public information: the book snapshot, the anonymous trade
tape (``Market.tape``), and their own account and orders.

Agents
    RandomAgent        zero-intelligence noise trader
    MarketMaker        Avellaneda-Stoikov style quoter, optional flow detection
    FundamentalTrader  trades toward a noisy private estimate of a hidden value
    MomentumTrader     trend follower on fast/slow moving averages

Helpers
    FlowDetector       flags one-sided aggressive flow from the public tape
    FundamentalValue   the hidden value process FundamentalTraders observe
    run_step           advances the simulation one step, agents in random order
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from order_book import OrderBook, Report


# --------------------------------------------------------------------------- #
# Accounting and routing
# --------------------------------------------------------------------------- #


@dataclass
class Account:
    cash: int
    shares: int
    open_since: dict[int, int] = field(default_factory=dict)  # order_id -> step


@dataclass(frozen=True)
class PublicTrade:
    """One fill as the public tape shows it. No client identities."""

    step: int
    price: int
    quantity: int
    aggressor: str  # "buy" or "sell": the side of the order that crossed
    taker_seq: int  # shared by every fill of one aggressive order


class Market:
    """Routes agent actions to the book and settles cash/shares for both parties."""

    def __init__(self, book: OrderBook, tape_length: int = 50_000):
        self.book = book
        self.accounts: dict[int, Account] = {}
        self.step = 0
        self.last_price: int | None = None
        self.tape: deque[PublicTrade] = deque(maxlen=tape_length)
        # Monitoring: orders dropped for lack of funds or for falling outside the window.
        self.budget_rejects = {"bid": 0, "ask": 0}
        self.window_rejects = {"bid": 0, "ask": 0}
        self._taker_seq = 0

    # ----- read-only views ------------------------------------------------- #

    def open_orders(self, cid: int, side: str | None = None) -> list[dict]:
        return [o for o in self.book.get_client_orders(cid) if side in (None, o["side"])]

    def free_cash(self, cid: int) -> int:
        return self.accounts[cid].cash - sum(o["price"] * o["quantity"] for o in self.open_orders(cid, "bid"))

    def free_shares(self, cid: int) -> int:
        return self.accounts[cid].shares - sum(o["quantity"] for o in self.open_orders(cid, "ask"))

    def best_bid(self) -> int | None:
        return max((p for p, v, _ in self.book.get_snapshot()["bids"] if v > 0), default=None)

    def best_ask(self) -> int | None:
        return min((p for p, v, _ in self.book.get_snapshot()["asks"] if v > 0), default=None)

    def reference_price(self) -> int:
        mid = self.book.mid_price
        if mid is not None:
            return int(round(mid))
        return self.last_price or self.book.initial_mid_price

    # ----- settlement ------------------------------------------------------ #

    def _settle(self, report: Report, side: str) -> Report:
        trades = report["trades"]
        if trades:
            self._taker_seq += 1
        aggressor = "buy" if side == "bid" else "sell"
        for t in trades:
            notional = t["price"] * t["quantity"]
            buyer, seller = self.accounts[t["buyer_id"]], self.accounts[t["seller_id"]]
            buyer.cash -= notional
            buyer.shares += t["quantity"]
            seller.cash += notional
            seller.shares -= t["quantity"]
            self.last_price = t["price"]
            self.tape.append(PublicTrade(self.step, t["price"], t["quantity"], aggressor, self._taker_seq))
        if report["resting_quantity"]:
            self.accounts[report["client_id"]].open_since[report["order_id"]] = self.step
        return report

    # ----- actions --------------------------------------------------------- #

    def limit(self, cid: int, side: str, price: int, qty: int) -> Report | None:
        """Place a limit order. Orders outside the window are dropped, never moved."""
        if price < 1 or qty < 1:
            return None
        lo_bid, hi_ask = int(self.book.bid_prices[1]), int(self.book.ask_prices[1])
        if (side == "bid" and price < lo_bid) or (side == "ask" and price > hi_ask):
            self.window_rejects[side] += 1
            return None
        cap = self.free_cash(cid) // price if side == "bid" else self.free_shares(cid)
        qty = min(qty, cap)
        if qty < 1:
            self.budget_rejects[side] += 1
            return None
        place = self.book.place_bid if side == "bid" else self.book.place_ask
        try:
            place(cid, price, qty)
        except ValueError:  # the book's own window check, e.g. at the boundary tick
            self.window_rejects[side] += 1
            return None
        return self._settle(self.book.last_order_report, side)

    def ioc(self, cid: int, side: str, price: int, qty: int) -> Report | None:
        """Immediate-or-cancel: trade up to ``price`` now, cancel whatever would rest."""
        report = self.limit(cid, side, price, qty)
        if report is not None and report["resting_quantity"]:
            self.cancel(cid, report["order_id"])
        return report

    def market(self, cid: int, side: str, qty: int) -> Report | None:
        if side == "bid":  # walk the asks conservatively so the buy is always funded
            cash, affordable = self.free_cash(cid), 0
            for price, volume, _ in self.book.get_snapshot()["asks"]:
                take = min(volume, cash // price, qty - affordable)
                affordable, cash = affordable + take, cash - take * price
                if take < volume:
                    break
            qty = affordable
        else:
            qty = min(qty, self.free_shares(cid))
        if qty < 1:
            self.budget_rejects[side] += 1
            return None
        report = self.book.buy(cid, qty) if side == "bid" else self.book.sell(cid, qty)
        return self._settle(report, side)

    def cancel(self, cid: int, order_id: int) -> None:
        self.book.cancel_order(order_id, cid)
        self.accounts[cid].open_since.pop(order_id, None)

    def cancel_all(self, cid: int) -> None:
        for o in self.open_orders(cid):
            self.book.cancel_order(o["order_id"], cid)
        self.accounts[cid].open_since.clear()


def filled_quantity(report: Report | None) -> int:
    return sum(t["quantity"] for t in report["trades"]) if report else 0


def average_price(report: Report | None) -> float | None:
    qty = filled_quantity(report)
    if not qty:
        return None
    return sum(t["price"] * t["quantity"] for t in report["trades"]) / qty


# --------------------------------------------------------------------------- #
# Shared processes
# --------------------------------------------------------------------------- #


class FundamentalValue:
    """Hidden value in ticks. Only FundamentalTraders see it, and only through noise.

    Each step: value += reversion * (anchor - value) + sigma * N(0, 1).
    sigma = 0 gives a constant value; reversion = 0 gives a random walk.
    This drives beliefs, not prices: the price still comes from the order flow.
    """

    def __init__(self, rng: np.random.Generator, initial: float, sigma: float = 0.3,
                 reversion: float = 0.0, anchor: float | None = None):
        self.rng = rng
        self.value = float(initial)
        self.sigma, self.reversion = sigma, reversion
        self.anchor = float(initial) if anchor is None else float(anchor)
        self.history = [self.value]

    def advance(self) -> None:
        self.value += self.reversion * (self.anchor - self.value) + self.sigma * self.rng.standard_normal()
        self.history.append(self.value)


class FlowDetector:
    """Flags persistent one-sided aggressive flow, using the public tape only.

    ``signal(market)`` returns a value in [-1, 1]: positive means persistent
    buying pressure, negative persistent selling, 0 nothing unusual.

    imbalance    (buy - sell) aggressive volume over the last ``window`` steps
    persistence  that imbalance averaged over the last ``memory`` steps
    regularity   share of same-side aggressive orders that have the most common
                 size (a TWAP fingerprint); it amplifies an existing signal

    One detector can be shared by several market makers; it updates once per step.
    ``history`` holds (step, imbalance, regularity, signal) for measuring
    detection delay and false positives.
    """

    def __init__(self, window: int = 20, memory: int = 10, threshold: float = 0.3,
                 regularity_boost: float = 0.5, min_orders: int = 5):
        self.window, self.threshold = window, threshold
        self.regularity_boost, self.min_orders = regularity_boost, min_orders
        self._imbalances: deque[float] = deque(maxlen=memory)
        self._last_step: int | None = None
        self._signal = 0.0
        self.history: list[tuple[int, float, float, float]] = []

    def signal(self, market: Market) -> float:
        if market.step == self._last_step:
            return self._signal
        self._last_step = market.step

        start = market.step - self.window
        volume = {"buy": 0, "sell": 0}
        order_sizes: dict[str, dict[int, int]] = {"buy": {}, "sell": {}}
        for t in reversed(market.tape):
            if t.step <= start:
                break
            volume[t.aggressor] += t.quantity
            sizes = order_sizes[t.aggressor]
            sizes[t.taker_seq] = sizes.get(t.taker_seq, 0) + t.quantity

        total = volume["buy"] + volume["sell"]
        imbalance = (volume["buy"] - volume["sell"]) / total if total else 0.0
        self._imbalances.append(imbalance)
        persistent = float(np.mean(self._imbalances))

        sizes = list(order_sizes["buy" if persistent > 0 else "sell"].values())
        regularity = 0.0
        if len(sizes) >= self.min_orders:
            regularity = np.unique(sizes, return_counts=True)[1].max() / len(sizes)

        strength = max(abs(persistent) - self.threshold, 0.0) / (1.0 - self.threshold)
        strength *= 1.0 + self.regularity_boost * regularity
        self._signal = math.copysign(min(strength, 1.0), persistent) if strength > 0 else 0.0
        self.history.append((market.step, imbalance, regularity, self._signal))
        return self._signal


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #


class Agent:
    def __init__(self, cid: int, market: Market, rng: np.random.Generator, cash: int, shares: int):
        self.cid, self.market, self.rng = cid, market, rng
        market.accounts[cid] = Account(cash, shares)

    @property
    def account(self) -> Account:
        return self.market.accounts[self.cid]

    def sync(self) -> list[dict]:
        """Live orders, after dropping filled or evicted ids from ``open_since``."""
        orders = self.market.open_orders(self.cid)
        live = {o["order_id"] for o in orders}
        acc = self.account
        acc.open_since = {i: s for i, s in acc.open_since.items() if i in live}
        return orders

    def random_cancels(self, base: float, per_tick: float) -> None:
        """Cancel each resting order with probability base + per_tick * distance from best."""
        m = self.market
        best = {"bid": m.best_bid(), "ask": m.best_ask()}
        for o in self.sync():
            ref = best[o["side"]]
            distance = 0 if ref is None else abs(ref - o["price"])
            if self.rng.random() < min(1.0, base + per_tick * distance):
                m.cancel(self.cid, o["order_id"])

    def act(self) -> None:
        raise NotImplementedError


class RandomAgent(Agent):
    """Zero-intelligence noise trader.

    Passive orders sit a geometric number of ticks behind the opposite best
    quote, so most land near the top of the book. Resting orders are cancelled
    with a per-step probability that grows with distance from the best price,
    which gives exponential lifetimes instead of a fixed time-to-live.

    ``balance_weight`` (0 to 1) tilts the side choice toward what the agent can
    afford: 0 is a fair coin, 1 picks bids in proportion to the cash share of
    the agent's free wealth. Check ``Market.budget_rejects`` before raising it.
    """

    def __init__(self, *args, activity: float = 0.6, market_prob: float = 0.15,
                 cross_prob: float = 0.05, max_qty: int = 10, mean_offset: float = 3.0,
                 cancel_prob: float = 0.02, cancel_per_tick: float = 0.01,
                 balance_weight: float = 0.0):
        super().__init__(*args)
        self.activity, self.market_prob, self.cross_prob = activity, market_prob, cross_prob
        self.max_qty, self.mean_offset = max_qty, mean_offset
        self.cancel_prob, self.cancel_per_tick = cancel_prob, cancel_per_tick
        self.balance_weight = balance_weight

    def _bid_probability(self) -> float:
        if self.balance_weight == 0:
            return 0.5
        m = self.market
        cash = max(m.free_cash(self.cid), 0)
        stock = max(m.free_shares(self.cid), 0) * m.reference_price()
        cash_share = cash / (cash + stock) if cash + stock else 0.5
        return (1 - self.balance_weight) * 0.5 + self.balance_weight * cash_share

    def act(self) -> None:
        m, rng = self.market, self.rng
        self.random_cancels(self.cancel_prob, self.cancel_per_tick)
        if rng.random() > self.activity:
            return
        side = "bid" if rng.random() < self._bid_probability() else "ask"
        qty = int(rng.integers(1, self.max_qty + 1))
        if rng.random() < self.market_prob:
            m.market(self.cid, side, qty)
            return

        ref = m.reference_price()
        best_bid, best_ask = m.best_bid(), m.best_ask()
        opposite = (best_ask if best_ask is not None else ref + 1) if side == "bid" else \
                   (best_bid if best_bid is not None else ref - 1)
        if rng.random() < self.cross_prob:
            # Marketable limit: at or one tick through the opposite best.
            through = int(rng.integers(0, 2))
            price = opposite + through if side == "bid" else opposite - through
        else:
            # Geometric offset: 0, 1, 2, ... ticks behind the opposite best, mean = mean_offset.
            offset = int(rng.geometric(1.0 / (self.mean_offset + 1.0))) - 1
            price = opposite - 1 - offset if side == "bid" else opposite + 1 + offset
        m.limit(self.cid, side, price, qty)


class MarketMaker(Agent):
    """Two-sided quoter in the spirit of Avellaneda-Stoikov, with optional flow detection.

    reservation = reference - inventory * gamma * var * horizon
    spread      = gamma * var * horizon + (2 / gamma) * ln(1 + gamma / kappa)

    ``var`` is the variance of one-step reference price changes (ticks^2),
    estimated over the last ``vol_window`` steps and floored at ``min_var``.
    ``kappa`` mostly sets the base spread (about 2 / kappa ticks when gamma is
    small); ``gamma`` sets how hard inventory skews the quotes.

    Quotes are post-only (clipped so they never cross other participants), and
    a side is only cancelled and replaced when its price changes or its size
    has run down, so the market maker keeps its queue position. The side that
    would push inventory past ``max_inventory`` is pulled.

    With a ``detector``, a signal s in [-1, 1] moves the reservation price
    ``detect_skew * s`` ticks toward the pressure, widens the pressured side by
    ``detect_widen * |s|`` ticks, and shrinks its size by ``detect_size_cut * |s|``.
    """

    def __init__(self, *args, gamma: float = 0.002, kappa: float = 0.5, horizon: int = 50,
                 vol_window: int = 50, min_var: float = 0.25, size: tuple[int, int] = (5, 20),
                 refill_frac: float = 0.5, target_shares: int | None = None,
                 max_inventory: int = 200, detector: FlowDetector | None = None,
                 detect_skew: float = 4.0, detect_widen: float = 3.0, detect_size_cut: float = 0.7):
        super().__init__(*args)
        self.gamma, self.kappa, self.horizon, self.min_var = gamma, kappa, horizon, min_var
        self.size, self.refill_frac, self.max_inventory = size, refill_frac, max_inventory
        self.target = self.account.shares if target_shares is None else target_shares
        self.detector = detector
        self.detect_skew, self.detect_widen, self.detect_size_cut = detect_skew, detect_widen, detect_size_cut
        self._prices: deque[int] = deque(maxlen=vol_window + 1)
        self._base_size = {"bid": self._draw_size(), "ask": self._draw_size()}
        self.last_signal = 0.0

    def _draw_size(self) -> int:
        return int(self.rng.integers(self.size[0], self.size[1] + 1))

    def _variance(self) -> float:
        if len(self._prices) < 3:
            return self.min_var
        changes = np.diff(np.asarray(self._prices, dtype=float))
        return max(float(np.mean(changes ** 2)), self.min_var)

    def _others_best(self, side: str) -> int | None:
        """Best price on ``side`` from everyone except this market maker."""
        own: dict[int, int] = {}
        for o in self.market.open_orders(self.cid, side):
            own[o["price"]] = own.get(o["price"], 0) + o["quantity"]
        levels = self.market.book.get_snapshot()["bids" if side == "bid" else "asks"]
        prices = [p for p, v, _ in levels if v - own.get(p, 0) > 0]
        if not prices:
            return None
        return max(prices) if side == "bid" else min(prices)

    def _requote(self, side: str, price: int, scale: float, allowed: bool) -> None:
        m = self.market
        current = m.open_orders(self.cid, side)
        target_qty = max(1, round(self._base_size[side] * scale))
        if allowed and len(current) == 1 and current[0]["price"] == price \
                and self.refill_frac * target_qty <= current[0]["quantity"] <= target_qty:
            return  # same price, enough size left: keep queue position
        for o in current:
            m.cancel(self.cid, o["order_id"])
        if allowed:
            self._base_size[side] = self._draw_size()
            m.limit(self.cid, side, price, max(1, round(self._base_size[side] * scale)))

    def act(self) -> None:
        m = self.market
        ref = m.reference_price()
        self._prices.append(ref)
        var = self._variance()
        inventory = self.account.shares - self.target

        reservation = ref - inventory * self.gamma * var * self.horizon
        spread = self.gamma * var * self.horizon + (2.0 / self.gamma) * math.log1p(self.gamma / self.kappa)
        half = max(spread / 2.0, 1.0)

        widen = {"bid": 0.0, "ask": 0.0}
        scale = {"bid": 1.0, "ask": 1.0}
        signal = self.detector.signal(m) if self.detector is not None else 0.0
        self.last_signal = signal
        if signal:
            pressured = "ask" if signal > 0 else "bid"  # buyers hit asks, sellers hit bids
            reservation += self.detect_skew * signal
            widen[pressured] = self.detect_widen * abs(signal)
            scale[pressured] = max(1.0 - self.detect_size_cut * abs(signal), 0.0)

        bid = int(round(reservation - half - widen["bid"]))
        ask = int(round(reservation + half + widen["ask"]))
        ask = max(ask, bid + 2)

        # Post-only: never cross anyone else's resting order.
        others_ask, others_bid = self._others_best("ask"), self._others_best("bid")
        if others_ask is not None:
            bid = min(bid, others_ask - 1)
        if others_bid is not None:
            ask = max(ask, others_bid + 1)

        self._requote("bid", bid, scale["bid"], inventory < self.max_inventory and scale["bid"] > 0)
        self._requote("ask", ask, scale["ask"], inventory > -self.max_inventory and scale["ask"] > 0)


class FundamentalTrader(Agent):
    """Trades toward a noisy private estimate of a hidden fundamental value.

    belief = value + private bias (fixed per agent) + observation noise (per step).
    When the reference price is more than ``threshold`` ticks from the belief,
    the agent trades toward it: a passive limit for a modest gap, an
    immediate-or-cancel order capped at the belief for a gap of at least
    ``aggressive_gap``. Size grows with the gap, up to ``max_qty``.

    These agents anchor the price and make it recover after an execution
    pushes it, which separates temporary from permanent impact.
    """

    def __init__(self, *args, fundamental: FundamentalValue, belief_bias_sd: float = 2.0,
                 obs_noise: float = 1.0, activity: float = 0.2, threshold: float = 2.0,
                 aggressive_gap: float = 8.0, size_per_tick: float = 1.0, max_qty: int = 20,
                 cancel_prob: float = 0.05):
        super().__init__(*args)
        self.fundamental = fundamental
        self.bias = float(self.rng.normal(0.0, belief_bias_sd))
        self.obs_noise, self.activity, self.threshold = obs_noise, activity, threshold
        self.aggressive_gap, self.size_per_tick, self.max_qty = aggressive_gap, size_per_tick, max_qty
        self.cancel_prob = cancel_prob

    def act(self) -> None:
        m, rng = self.market, self.rng
        belief = self.fundamental.value + self.bias + rng.normal(0.0, self.obs_noise)

        # Drop orders the agent no longer believes in, plus a few at random.
        for o in self.sync():
            stale = o["price"] > belief if o["side"] == "bid" else o["price"] < belief
            if stale or rng.random() < self.cancel_prob:
                m.cancel(self.cid, o["order_id"])

        if rng.random() > self.activity:
            return
        gap = belief - m.reference_price()
        if abs(gap) < self.threshold:
            return
        side = "bid" if gap > 0 else "ask"
        qty = min(self.max_qty, max(1, round(self.size_per_tick * (abs(gap) - self.threshold))))

        if abs(gap) >= self.aggressive_gap:
            # Take liquidity, but never pay more (or sell for less) than the belief.
            price = math.floor(belief) if side == "bid" else math.ceil(belief)
            m.ioc(self.cid, side, price, qty)
            return

        if side == "bid":
            price, best_ask = math.floor(belief - self.threshold), m.best_ask()
            if best_ask is not None:
                price = min(price, best_ask - 1)
        else:
            price, best_bid = math.ceil(belief + self.threshold), m.best_bid()
            if best_bid is not None:
                price = max(price, best_bid + 1)
        m.limit(self.cid, side, price, qty)


class MomentumTrader(Agent):
    """Trend follower on fast and slow exponential moving averages of the price.

    Enters with a market order when fast - slow exceeds ``threshold`` ticks
    (buy on an uptrend, sell on a downtrend), sized by the signal up to
    ``max_qty``. Exits the whole position after ``hold_steps``, when the trend
    reverses past the threshold, or after a ``stop_loss`` tick adverse move.
    One position at a time; the no-short rule caps sells at shares held.
    """

    def __init__(self, *args, fast: int = 5, slow: int = 30, threshold: float = 1.0,
                 activity: float = 0.3, size_per_tick: float = 2.0, max_qty: int = 15,
                 hold_steps: int = 40, stop_loss: float = 10.0):
        super().__init__(*args)
        self.alpha_fast, self.alpha_slow, self.slow = 2.0 / (fast + 1), 2.0 / (slow + 1), slow
        self.threshold, self.activity = threshold, activity
        self.size_per_tick, self.max_qty = size_per_tick, max_qty
        self.hold_steps, self.stop_loss = hold_steps, stop_loss
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.seen = 0
        self.position: tuple[str, int, int, float] | None = None  # side, qty, entry step, entry price

    def act(self) -> None:
        m, rng = self.market, self.rng
        price = m.reference_price()
        if self.ema_fast is None:
            self.ema_fast = self.ema_slow = float(price)
        else:
            self.ema_fast += self.alpha_fast * (price - self.ema_fast)
            self.ema_slow += self.alpha_slow * (price - self.ema_slow)
        self.seen += 1
        signal = self.ema_fast - self.ema_slow

        if self.position is not None:
            side, qty, since, entry = self.position
            direction = 1 if side == "bid" else -1
            if (m.step - since >= self.hold_steps
                    or direction * signal < -self.threshold
                    or direction * (price - entry) <= -self.stop_loss):
                report = m.market(self.cid, "ask" if side == "bid" else "bid", qty)
                remaining = qty - filled_quantity(report)
                self.position = None if remaining <= 0 else (side, remaining, since, entry)
            return

        if self.seen < self.slow or rng.random() > self.activity or abs(signal) < self.threshold:
            return
        side = "bid" if signal > 0 else "ask"
        qty = min(self.max_qty, max(1, round(self.size_per_tick * (abs(signal) - self.threshold))))
        report = m.market(self.cid, side, qty)
        filled = filled_quantity(report)
        if filled:
            self.position = (side, filled, m.step, average_price(report))


# --------------------------------------------------------------------------- #
# Simulation step
# --------------------------------------------------------------------------- #


def run_step(market: Market, agents: list[Agent], rng: np.random.Generator,
             fundamental: FundamentalValue | None = None) -> None:
    """Advance one step: move the hidden value, then every agent acts once in random order.

    Shuffling stops whoever is first in the list from always acting first.
    """
    if fundamental is not None:
        fundamental.advance()
    for i in rng.permutation(len(agents)):
        agents[i].act()
    market.step += 1