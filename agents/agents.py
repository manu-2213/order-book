"""Background-market participants for the limit order book.

Cost scales with events, not with the number of notional traders:

    NoiseFlow     Poisson order arrivals (rate x attention) from one pooled
                  account; sides, sizes and offsets drawn vectorised
    Speculators   one array-backed population of fundamentalists and chartists
                  with Brock-Hommes strategy switching; only the sampled
                  active members do any work in a step
    Institution   occasionally executes a large metaorder as regular TWAP
                  child orders; returns immediately while idle
    MarketMaker   Avellaneda-Stoikov quoter reading PublicState, optional
                  flow detection; requotes only when its prices move

Agents only use public information (``Market.public``, the book's best
quotes/levels) plus their own accounts and orders.
"""

import math

import numpy as np

from agents.market import Market


class FundamentalValue:
    """Hidden value in ticks: value += reversion * (anchor - value) + sigma * N(0, 1).

    Only speculators see it, and only through private noise. It drives
    beliefs, not prices; the price still comes from the order flow.
    """

    def __init__(self, rng: np.random.Generator, initial: float, sigma: float = 0.3,
                 reversion: float = 0.0, record: bool = True):
        self.rng, self.value, self.sigma, self.reversion = rng, float(initial), sigma, reversion
        self.anchor = float(initial)
        self.history: list[float] | None = [self.value] if record else None

    def advance(self) -> None:
        self.value += self.reversion * (self.anchor - self.value) + self.sigma * self.rng.standard_normal()
        if self.history is not None:
            self.history.append(self.value)


class Agent:
    def __init__(self, cid: int, market: Market, rng: np.random.Generator, cash: int, shares: int):
        self.cid, self.market, self.rng = cid, market, rng
        self.account = market.open_account(cid, cash, shares)

    def act(self) -> None:
        raise NotImplementedError


class NoiseFlow(Agent):
    """Zero-intelligence order flow from one pooled account.

    Each step: cancel a Binomial(resting, cancel_prob) random subset of resting
    orders, then submit Poisson(rate * attention) new orders of 1..max_qty
    lots. A fraction ``market_frac`` are market orders (half size, so takers
    are smaller than makers as in real books), ``cross_frac`` are marketable
    limits at the opposite best, the rest rest Geometric ticks behind it
    (mean ``mean_offset`` + 1). The bid probability is tilted by
    ``balance_weight`` toward the pool's cash share of wealth, so the pool
    cannot run itself out of cash or stock in a long trend.
    """

    def __init__(self, *args, rate: float = 2.5, cancel_prob: float = 0.015, max_qty: int = 12,
                 mean_offset: float = 3.0, market_frac: float = 0.15, cross_frac: float = 0.05,
                 balance_weight: float = 0.5):
        super().__init__(*args)
        self.rate, self.cancel_prob, self.max_qty = rate, cancel_prob, max_qty
        self.p_geometric, self.market_frac, self.cross_frac = 1 / (mean_offset + 1), market_frac, cross_frac
        self.balance_weight = balance_weight

    def act(self) -> None:
        m, rng, book, cid = self.market, self.rng, self.market.book, self.cid
        ids = book.client_order_ids(cid)
        if ids:
            k = rng.binomial(len(ids), self.cancel_prob)
            if k:
                for order_id in rng.choice(list(ids), k, replace=False):
                    m.cancel(cid, int(order_id))
        n = rng.poisson(self.rate * m.public.attention)
        if not n:
            return
        account, mid = self.account, m.public.mid
        cash_share = account.cash / (account.cash + account.shares * mid)
        bids = rng.random(n) < 0.5 + self.balance_weight * (cash_share - 0.5)
        sizes = rng.integers(1, self.max_qty + 1, n)
        kinds = rng.random(n)
        offsets = rng.geometric(self.p_geometric, n)
        cross = self.market_frac + self.cross_frac
        for i in range(n):
            side = "bid" if bids[i] else "ask"
            quantity = int(sizes[i])
            if kinds[i] < self.market_frac:
                m.market(cid, side, (quantity + 1) // 2)
                continue
            bid, ask = book.best_bid, book.best_ask
            if side == "bid":
                opposite = ask if ask is not None else round(m.public.mid) + 1
                price = opposite if kinds[i] < cross else opposite - int(offsets[i])
            else:
                opposite = bid if bid is not None else round(m.public.mid) - 1
                price = opposite if kinds[i] < cross else opposite + int(offsets[i])
            m.limit(cid, side, price, quantity)


class Speculators:
    """Array-backed population; each member is a fundamentalist or a chartist.

    Fundamentalists trade toward value + private bias + fresh observation
    noise: a passive limit for a modest gap, a market order for a gap of at
    least ``aggressive_gap``; sizes grow with the gap up to ``max_qty``. Chartists follow the momentum indicator
    of their assigned lookback pair with market orders, exiting in chunks of at
    most ``max_qty`` when the trend reverses or fades. Positions are capped at
    ``max_position`` either way. Every ``switch_every`` steps each member re-picks its
    mode with logit probability exp(beta * fitness) (Brock-Hommes), where
    fitness is an EWMA of the mode's mean wealth return over the block.
    """

    def __init__(self, first_cid: int, market: Market, rng: np.random.Generator,
                 fundamental: FundamentalValue, n: int = 40, cash: int = 2_000_000,
                 shares: int = 200, p_act: float = 0.05, bias_sd: float = 3.0,
                 obs_noise: float = 1.0, threshold: float = 2.0, aggressive_gap: float = 8.0,
                 max_qty: int = 20, max_position: int | None = None, chart_threshold: float = 3.0,
                 chart_max_qty: int = 10, size_per_tick: float = 2.0,
                 switch_every: int = 50, beta: float = 2.0, fitness_memory: float = 0.1):
        self.market, self.rng, self.fundamental, self.n = market, rng, fundamental, n
        self.cids = list(range(first_cid, first_cid + n))
        self.accounts = [market.open_account(cid, cash, shares) for cid in self.cids]
        self.target_shares = shares
        self.max_position = shares if max_position is None else max_position
        self.p_act, self.obs_noise, self.threshold = p_act, obs_noise, threshold
        self.aggressive_gap, self.max_qty = aggressive_gap, max_qty
        self.chart_threshold, self.chart_max_qty, self.size_per_tick = chart_threshold, chart_max_qty, size_per_tick
        self.switch_every, self.beta, self.fitness_memory = switch_every, beta, fitness_memory
        self.bias = rng.normal(0.0, bias_sd, n)
        self.mode = rng.integers(0, 2, n)  # 0 fundamentalist, 1 chartist
        self.lookback = rng.integers(0, len(market.public.lookbacks), n)
        self.fitness = np.zeros(2)
        self._wealth = self.wealth()

    def wealth(self) -> np.ndarray:
        mid = self.market.public.mid
        return np.array([a.cash + a.shares * mid for a in self.accounts])

    def act(self) -> None:
        for i in np.flatnonzero(self.rng.random(self.n) < self.p_act):
            self._act_one(int(i))
        if self.market.step % self.switch_every == self.switch_every - 1:
            self._switch()

    def _act_one(self, i: int) -> None:
        m, cid, account = self.market, self.cids[i], self.accounts[i]
        mid = m.public.mid
        position = account.shares - self.target_shares
        if m.book.client_order_ids(cid):
            m.cancel_all(cid)  # re-evaluate from scratch; stale quotes are not held
        if self.mode[i] == 0:
            belief = self.fundamental.value + self.bias[i] + self.rng.normal(0.0, self.obs_noise)
            gap = belief - mid
            if abs(gap) < self.threshold:
                return
            side = "bid" if gap > 0 else "ask"
            quantity = min(self.max_qty, max(1, round(abs(gap) - self.threshold)))
            quantity = min(quantity, self.max_position - position if side == "bid" else self.max_position + position)
            if quantity < 1:
                return
            if abs(gap) >= self.aggressive_gap:
                m.market(cid, side, quantity)
            elif side == "bid":
                ask = m.book.best_ask
                price = math.floor(belief - self.threshold)
                m.limit(cid, side, min(price, ask - 1) if ask is not None else price, quantity)
            else:
                bid = m.book.best_bid
                price = math.ceil(belief + self.threshold)
                m.limit(cid, side, max(price, bid + 1) if bid is not None else price, quantity)
            return
        signal = m.public.momentum[self.lookback[i]]
        if position and (position * signal < 0 or abs(signal) < 0.25 * self.chart_threshold):
            m.market(cid, "ask" if position > 0 else "bid", min(abs(position), self.chart_max_qty))
            return
        if abs(signal) < self.chart_threshold:
            return
        side = "bid" if signal > 0 else "ask"
        quantity = min(self.chart_max_qty, max(1, round(self.size_per_tick * (abs(signal) - self.chart_threshold))))
        quantity = min(quantity, self.max_position - position if side == "bid" else self.max_position + position)
        if quantity >= 1:
            m.market(cid, side, quantity)

    def _switch(self) -> None:
        wealth = self.wealth()
        returns = (wealth - self._wealth) / self._wealth * 100
        self._wealth = wealth
        for mode in (0, 1):
            members = self.mode == mode
            if members.any():
                self.fitness[mode] += self.fitness_memory * (returns[members].mean() - self.fitness[mode])
        p_chartist = 1 / (1 + math.exp(-self.beta * (self.fitness[1] - self.fitness[0])))
        self.mode = (self.rng.random(self.n) < p_chartist).astype(int)


class Institution(Agent):
    """Executes rare, large metaorders as evenly sized market child orders.

    With probability ``start_prob`` per idle step it draws a side, a total size
    and a duration, then sends one child every ``interval`` steps. Regular
    child sizes and persistent one-sided flow are what detectors look for.
    """

    def __init__(self, *args, start_prob: float = 0.002, size=(200, 600),
                 duration=(100, 300), interval: int = 5):
        super().__init__(*args)
        self.start_prob, self.size, self.duration, self.interval = start_prob, size, duration, interval
        self.remaining = 0
        self.side = "bid"
        self.child = self.next_step = 0

    def act(self) -> None:
        m, rng = self.market, self.rng
        if not self.remaining:
            if rng.random() >= self.start_prob:
                return
            self.side = "bid" if rng.random() < 0.5 else "ask"
            total = int(rng.integers(*self.size))
            self.child = max(1, round(total * self.interval / int(rng.integers(*self.duration))))
            self.remaining, self.next_step = total, m.step
        if m.step < self.next_step:
            return
        report = m.market(self.cid, self.side, min(self.child, self.remaining))
        self.remaining = self.remaining - report["filled_quantity"] if report else 0
        self.next_step = m.step + self.interval


class MarketMaker(Agent):
    """Two-sided post-only quoter in the spirit of Avellaneda-Stoikov.

    reservation = mid - inventory * gamma * var * horizon
    spread      = gamma * var * horizon + (2 / gamma) * ln(1 + gamma / kappa)

    ``var`` is the fast EWMA variance of mid changes from PublicState, floored
    at ``min_var``; the inventory skew and the half-spread are each capped at
    ``max_skew`` ticks so a volatility spike cannot make the quoter cross the
    book. A side is only cancelled and replaced when its target price moves at
    least one tick from the resting quote or its size has run below
    ``refill_frac`` of target, so the quoter keeps queue position. The side that would push inventory past
    ``max_inventory`` is pulled. With ``detect``, the public flow signal s in
    [-1, 1] skews the reservation ``detect_skew * s`` ticks, widens the
    pressured side by ``detect_widen * |s|`` and cuts its size by ``detect_size_cut * |s|``.
    """

    def __init__(self, *args, gamma: float = 0.002, kappa: float = 1.0, horizon: int = 50,
                 min_var: float = 0.25, max_skew: float = 10.0, size=(5, 20),
                 refill_frac: float = 0.5, max_inventory: int = 200, detect: bool = False, detect_skew: float = 4.0,
                 detect_widen: float = 3.0, detect_size_cut: float = 0.7):
        super().__init__(*args)
        self.gamma, self.kappa, self.horizon = gamma, kappa, horizon
        self.min_var, self.max_skew = min_var, max_skew
        self.size, self.refill_frac, self.max_inventory = size, refill_frac, max_inventory
        self.target = self.account.shares
        self.detect, self.detect_skew = detect, detect_skew
        self.detect_widen, self.detect_size_cut = detect_widen, detect_size_cut
        self.base_size = {"bid": self._draw_size(), "ask": self._draw_size()}
        self.quotes: dict[str, int | None] = {"bid": None, "ask": None}  # own order id per side

    def _draw_size(self) -> int:
        return int(self.rng.integers(self.size[0], self.size[1] + 1))

    def _requote(self, side: str, order, target_price: float, scale: float, allowed: bool) -> None:
        m = self.market
        target = max(1, round(self.base_size[side] * scale))
        if order is not None:
            if (allowed and abs(order["price"] - target_price) < 1.0
                    and self.refill_frac * target <= order["quantity"] <= target):
                return
            m.cancel(self.cid, order["order_id"])
        self.quotes[side] = None
        if allowed:
            self.base_size[side] = self._draw_size()
            report = m.limit(self.cid, side, round(target_price), max(1, round(self.base_size[side] * scale)))
            if report is not None and report["resting_quantity"]:
                self.quotes[side] = report["order_id"]

    def act(self) -> None:
        m, ps, book = self.market, self.market.public, self.market.book
        inventory = self.account.shares - self.target
        var = max(ps.var_fast, self.min_var)
        risk = self.gamma * var * self.horizon
        reservation = ps.mid - max(-self.max_skew, min(self.max_skew, inventory * risk))
        half = min(max((risk + (2 / self.gamma) * math.log1p(self.gamma / self.kappa)) / 2, 1.0), self.max_skew)
        widen = {"bid": 0.0, "ask": 0.0}
        scale = {"bid": 1.0, "ask": 1.0}
        signal = ps.flow_signal if self.detect else 0.0
        if signal:
            pressured = "ask" if signal > 0 else "bid"  # buyers hit asks, sellers hit bids
            reservation += self.detect_skew * signal
            widen[pressured] = self.detect_widen * abs(signal)
            scale[pressured] = max(1.0 - self.detect_size_cut * abs(signal), 0.0)
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
        self._requote("bid", own["bid"], bid, scale["bid"], inventory < self.max_inventory and scale["bid"] > 0)
        self._requote("ask", own["ask"], ask, scale["ask"], inventory > -self.max_inventory and scale["ask"] > 0)


def run_step(market: Market, agents: list, rng: np.random.Generator,
             fundamental: FundamentalValue | None = None) -> None:
    """Advance one step: move the hidden value, agents act in random order, publish state."""
    if fundamental is not None:
        fundamental.advance()
    for i in rng.permutation(len(agents)):
        agents[i].act()
    market.end_step()
