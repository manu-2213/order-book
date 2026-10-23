"""Array-backed population of fundamentalists and chartists with strategy switching."""

import math

import numpy as np

from order_book.agents.fundamental import FundamentalValue
from order_book.config import SpeculatorConfig
from order_book.market import Market


class Speculators:
    """One population object; each member is a fundamentalist or a chartist.

    Fundamentalists trade toward value + private bias + fresh observation
    noise: a passive limit for a modest gap, a market order for a gap of at
    least ``aggressive_gap``; sizes grow with the gap up to ``max_qty``.
    Chartists follow the momentum indicator of their assigned lookback pair
    with market orders, exiting in chunks of at most ``chart_max_qty`` when
    the trend reverses or fades. Positions are capped at ``max_position``
    either way. Every ``switch_every`` steps each member re-picks its mode
    with logit probability exp(beta * fitness) (Brock-Hommes), where fitness
    is an EWMA of the mode's mean wealth return over the block.

    Only the sampled active members (``p_act`` each) do any work in a step.

    With ``fundamental.cfg.private`` each member holds a private valuation
    v_i += adapt * (mid - v_i) + sigma * N(0, 1), updated for all members once
    per step, and fundamentalists trade toward it instead of the shared value.
    """

    def __init__(self, first_cid: int, market: Market, rng: np.random.Generator,
                 fundamental: FundamentalValue, n: int, cash: int, shares: int,
                 cfg: SpeculatorConfig | None = None):
        cfg = self.cfg = cfg or SpeculatorConfig()
        self.market, self.rng, self.fundamental, self.n = market, rng, fundamental, n
        self.cids = list(range(first_cid, first_cid + n))
        self.accounts = [market.open_account(cid, cash, shares) for cid in self.cids]
        self.target_shares = shares
        self.max_position = shares if cfg.max_position is None else cfg.max_position
        self.bias = rng.normal(0.0, cfg.bias_sd, n)
        self.mode = rng.integers(0, 2, n)  # 0 fundamentalist, 1 chartist
        self.lookback = rng.integers(0, len(market.public.lookbacks), n)
        self.fitness = np.zeros(2)
        self.private = fundamental.cfg.private
        self.values = np.full(n, float(market.public.mid)) if self.private else None
        self._wealth = self.wealth()

    def wealth(self) -> np.ndarray:
        mid = self.market.public.mid
        return np.array([a.cash + a.shares * mid for a in self.accounts])

    def act(self) -> None:
        if self.private:
            fc = self.fundamental.cfg
            self.values += fc.adapt * (self.market.public.mid - self.values) + fc.sigma * self.rng.standard_normal(self.n)
        for i in np.flatnonzero(self.rng.random(self.n) < self.cfg.p_act):
            self._act_one(int(i))
        if self.market.step % self.cfg.switch_every == self.cfg.switch_every - 1:
            self._switch()

    def _act_one(self, i: int) -> None:
        m, cid, account, cfg = self.market, self.cids[i], self.accounts[i], self.cfg
        mid = m.public.mid
        position = account.shares - self.target_shares
        if m.book.client_order_ids(cid):
            m.cancel_all(cid)  # re-evaluate from scratch; stale quotes are not held
        if self.mode[i] == 0:
            value = self.values[i] if self.private else self.fundamental.value
            belief = value + self.bias[i] + self.rng.normal(0.0, cfg.obs_noise)
            gap = belief - mid
            if abs(gap) < cfg.threshold:
                return
            side = "bid" if gap > 0 else "ask"
            quantity = min(cfg.max_qty, max(1, round(abs(gap) - cfg.threshold)))
            quantity = min(quantity, self.max_position - position if side == "bid" else self.max_position + position)
            if quantity < 1:
                return
            if abs(gap) >= cfg.aggressive_gap:
                m.market(cid, side, quantity)
            elif side == "bid":
                ask = m.book.best_ask
                price = math.floor(belief - cfg.threshold)
                m.limit(cid, side, min(price, ask - 1) if ask is not None else price, quantity)
            else:
                bid = m.book.best_bid
                price = math.ceil(belief + cfg.threshold)
                m.limit(cid, side, max(price, bid + 1) if bid is not None else price, quantity)
            return
        signal = m.public.momentum[self.lookback[i]]
        if position and (position * signal < 0 or abs(signal) < 0.25 * cfg.chart_threshold):
            m.market(cid, "ask" if position > 0 else "bid", min(abs(position), cfg.chart_max_qty))
            return
        if abs(signal) < cfg.chart_threshold:
            return
        side = "bid" if signal > 0 else "ask"
        quantity = min(cfg.chart_max_qty, max(1, round(cfg.size_per_tick * (abs(signal) - cfg.chart_threshold))))
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
                self.fitness[mode] += self.cfg.fitness_memory * (returns[members].mean() - self.fitness[mode])
        p_chartist = 1 / (1 + math.exp(-self.cfg.beta * (self.fitness[1] - self.fitness[0])))
        self.mode = (self.rng.random(self.n) < p_chartist).astype(int)
