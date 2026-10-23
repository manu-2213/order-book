"""Institutional metaorders: the legitimate large orders that give an executor cover.

Each trader runs at most one metaorder at a time, drawn from a Poisson-like
start process with log-normal size, uniform participation rate and one of
five execution styles. Ground-truth labels (``MetaorderRecord``) go to a
``MetaorderLog`` that the simulator keeps in analysis mode only; no agent
reads them.
"""

import math
from dataclasses import dataclass

from order_book.agents.base import Agent
from order_book.config import InstitutionalConfig


@dataclass
class MetaorderRecord:
    """Ground truth for one metaorder; ``end``/``executed``/``notional`` are filled at completion."""

    id: int
    client_id: int
    side: str                 # "bid" buys, "ask" sells
    style: str
    total: int                # lots intended
    participation: float      # target share of market volume
    start: int                # step of the first child order
    planned_end: int          # start + implied duration
    arrival_mid: float
    end: int | None = None
    executed: int = 0
    notional: int = 0         # cash paid (buy) or received (sell)

    @property
    def avg_price(self) -> float | None:
        return self.notional / self.executed if self.executed else None

    @property
    def shortfall_bps(self) -> float | None:
        """Average fill price against the arrival mid, signed so that positive is a cost."""
        if not self.executed:
            return None
        sign = 1 if self.side == "bid" else -1
        return sign * (self.avg_price - self.arrival_mid) / self.arrival_mid * 1e4


class MetaorderLog:
    """Hands out ids and, in analysis mode, keeps every record. Agents only ``open`` their own."""

    def __init__(self, keep: bool):
        self.records: list[MetaorderRecord] | None = [] if keep else None
        self._next = 0

    def open(self, **fields) -> MetaorderRecord:
        record = MetaorderRecord(id=self._next, **fields)
        self._next += 1
        if self.records is not None:
            self.records.append(record)
        return record


class InstitutionalTrader(Agent):
    """Executes metaorders in one of ``EXECUTION_STYLES``; see ``InstitutionalConfig``.

    Progress is measured from the trader's own account (shares relative to
    the start of the metaorder), so passive fills need no callbacks. The side
    is a buy with probability 1/2 - tanh(inventory / endowment) / 2, where
    inventory is the holding relative to the initial endowment, so each
    trader's holding mean-reverts and the population's aggregate flow stays
    balanced over long runs instead of random-walking into the other agents'
    budgets. The size is capped by what the trader can fund, and whatever is
    left at the planned end is forced through by market order, as an
    execution desk would do.
    """

    def __init__(self, *args, cfg: InstitutionalConfig | None = None, log: MetaorderLog | None = None):
        super().__init__(*args)
        self.cfg = cfg or InstitutionalConfig()
        self.log = log or MetaorderLog(keep=False)
        self.record: MetaorderRecord | None = None
        self.endowment = self.account.shares
        self.cum_volume = 0  # own running sum of public per-step volume (for POV)
        self._style_p = None
        if self.cfg.style_weights is not None:
            total = sum(self.cfg.style_weights)
            self._style_p = [w / total for w in self.cfg.style_weights]

    # ----- lifecycle -------------------------------------------------------- #

    def act(self) -> None:
        m, ps = self.market, self.market.public
        self.cum_volume += ps.step_volume
        if self.record is None:
            if self.rng.random() < self.cfg.start_rate:
                self._start()
            return
        if m.step >= self.record.planned_end or self._remaining() <= 0:
            self._finish()
            return
        getattr(self, "_" + self.record.style)()

    def _start(self) -> None:
        rng, cfg, account = self.rng, self.cfg, self.account
        inventory = (account.shares - self.endowment) / max(self.endowment, 1)
        side = "bid" if rng.random() < 0.5 - 0.5 * math.tanh(cfg.inventory_pull * inventory) else "ask"
        size = int(min(cfg.size_max, round(cfg.size_median * math.exp(cfg.size_sigma * rng.standard_normal()))))
        participation = rng.uniform(*cfg.participation)
        style = cfg.styles[rng.choice(len(cfg.styles), p=self._style_p)]
        self.begin(side, size, participation, style)

    def begin(self, side: str, size: int, participation: float, style: str,
              duration: int | None = None) -> MetaorderRecord | None:
        """Start a metaorder now (also used to inject one from outside for impact experiments).

        The size is capped by what the account can fund; the duration defaults
        to size / (participation x expected market volume per step), clamped.
        """
        m, cfg, account, ps = self.market, self.cfg, self.account, self.market.public
        if self.record is not None:
            raise RuntimeError("a metaorder is already running")
        mid = ps.mid
        affordable = account.free_cash // math.ceil(mid * 1.05) if side == "bid" else account.free_shares
        size = min(int(size), int(affordable))
        if size < 1:
            return None
        if duration is None:
            volume = ps.volume_slow if ps.warmed_up else cfg.volume_fallback
            duration = int(min(cfg.duration[1], max(cfg.duration[0], size / (participation * max(volume, 1e-9)))))
        self.record = self.log.open(client_id=self.cid, side=side, style=style, total=size,
                                    participation=participation, start=m.step,
                                    planned_end=m.step + max(1, duration), arrival_mid=mid)
        self.start_shares, self.start_cash = account.shares, account.cash
        self.next_step, self.volume_at_start, self.resting = m.step, self.cum_volume, None
        getattr(self, "_" + style)()
        return self.record

    def _finish(self) -> None:
        m, rec = self.market, self.record
        if self.resting is not None:
            m.cancel_all(self.cid)
            self.resting = None
        remaining = self._remaining()
        if remaining > 0:
            m.market(self.cid, rec.side, remaining)
        rec.end, rec.executed = m.step, self._executed()
        rec.notional = abs(self.account.cash - self.start_cash)
        self.record = None

    # ----- progress ---------------------------------------------------------- #

    def _executed(self) -> int:
        delta = self.account.shares - self.start_shares
        return delta if self.record.side == "bid" else -delta

    def _remaining(self) -> int:
        return self.record.total - self._executed()

    def _slices_left(self) -> int:
        return max(1, math.ceil((self.record.planned_end - self.market.step) / self.cfg.interval))

    def _send(self, quantity: int) -> None:
        quantity = min(quantity, self._remaining())
        if quantity >= 1:
            self.market.market(self.cid, self.record.side, quantity)

    # ----- styles ------------------------------------------------------------ #

    def _twap_market(self) -> None:
        if self.market.step < self.next_step:
            return
        self._send(math.ceil(self._remaining() / self._slices_left()))
        self.next_step = self.market.step + self.cfg.interval

    def _twap_random(self) -> None:
        if self.market.step < self.next_step:
            return
        j = self.cfg.jitter
        child = math.ceil(self._remaining() / self._slices_left() * self.rng.uniform(1 - j, 1 + j))
        self._send(child)
        self.next_step = self.market.step + max(1, round(self.cfg.interval * self.rng.uniform(1 - j, 1 + j)))

    def _twap_passive(self) -> None:
        """Post each slice at own best; at the slice deadline cross the spread for what did not fill."""
        m, book, rec = self.market, self.market.book, self.record
        if m.step < self.next_step:
            return
        if self.resting is not None:
            m.cancel_all(self.cid)
            self.resting = None
            self._send(self.slice_target - self._executed())  # catch-up for the slice just ended
        slice_qty = math.ceil(self._remaining() / self._slices_left())
        self.slice_target = self._executed() + slice_qty
        if slice_qty >= 1:
            if rec.side == "bid":
                price = book.best_bid if book.best_bid is not None else math.floor(m.public.mid) - 1
            else:
                price = book.best_ask if book.best_ask is not None else math.ceil(m.public.mid) + 1
            report = m.limit(self.cid, rec.side, price, slice_qty)
            self.resting = report["order_id"] if report is not None and report["resting_quantity"] else None
        self.next_step = m.step + self.cfg.interval

    def _pov(self) -> None:
        """Track participation x market volume since the start; catch up over the final tenth."""
        m, rec = self.market, self.record
        if m.step < self.next_step:
            return
        horizon = rec.planned_end - rec.start
        if m.step >= rec.start + 0.9 * horizon:
            target = self._executed() + math.ceil(self._remaining() / self._slices_left())
        else:
            target = min(rec.total, math.floor(rec.participation * (self.cum_volume - self.volume_at_start)))
        self._send(target - self._executed())
        self.next_step = m.step + self.cfg.interval

    def _front_loaded(self) -> None:
        """Almgren-Chriss-like trajectory: remaining(t) = Q sinh(k (T - t)) / sinh(k T) with k T = front_load."""
        m, rec = self.market, self.record
        if m.step < self.next_step:
            return
        horizon = max(1, rec.planned_end - rec.start)
        elapsed = min(horizon, m.step - rec.start + self.cfg.interval)
        k = self.cfg.front_load / horizon
        scheduled_remaining = rec.total * math.sinh(k * (horizon - elapsed)) / math.sinh(k * horizon)
        self._send(math.ceil(self._remaining() - scheduled_remaining))
        self.next_step = m.step + self.cfg.interval
