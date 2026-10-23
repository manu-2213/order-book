"""The execution task (plan section 5.1) and the agent that carries a strategy through it.

``ExecutorAgent`` is an external participant (``Simulator.add_external``): it
acts before the background population with its own random stream, decides
every ``decision_interval`` steps by asking its ``Strategy`` for a
``Decision``, enforces the hard limits, and at the horizon forces whatever
is left through by market order. Its implementation shortfall includes the
forced completion in full.
"""

import math
from dataclasses import dataclass, field

from order_book.agents.base import Agent


@dataclass(frozen=True)
class ExecutionTask:
    """What the executor must do; the size is a fraction of expected market volume over the horizon."""

    side: str = "bid"                    # "bid" buys, "ask" sells
    size_fraction: float = 0.10          # Q = size_fraction x expected volume per step x horizon
    horizon: int = 2_000                 # T, steps
    decision_interval: int = 5           # N, steps between decisions
    max_resting_orders: int = 2          # hard limit on the executor's resting orders
    max_through_ticks: int = 5           # marketable limits may not go further through the opposite best
    max_participation: float | None = None  # per decision: at most this share of expected interval volume
    grace: int = 50                      # steps after the horizon to keep forcing if the book is thin


@dataclass
class Decision:
    """What a strategy wants at one decision point. Quantities in lots; ``limit_price`` in ticks."""

    cancel: bool = False                 # cancel the executor's resting orders first
    market_qty: int = 0                  # market order now
    limit_qty: int = 0                   # then a limit order at ``limit_price`` (None = own best)
    limit_price: int | None = None


@dataclass
class ExecState:
    """Everything a strategy may look at: public state plus the executor's own position."""

    step: int
    elapsed: int
    horizon: int
    decision_interval: int
    decisions_left: int
    side: str
    quantity: int
    remaining: int
    executed: int
    aggressive_executed: int
    arrival_mid: float
    mid: float
    best_bid: int | None
    best_ask: int | None
    spread: int
    imbalance: float
    public_volume_since_start: int
    own_resting: int
    own_resting_orders: list
    expected_volume_per_step: float
    own_flow_share: float                # own aggressive volume / all aggressive volume in the public window
    public: object                       # the PublicState itself, for anything else public

    @property
    def sign(self) -> int:
        return 1 if self.side == "bid" else -1

    def own_best(self) -> int:
        """Price at the executor's own side's best (one tick inside the mid if that side is empty)."""
        if self.side == "bid":
            return self.best_bid if self.best_bid is not None else math.floor(self.mid) - 1
        return self.best_ask if self.best_ask is not None else math.ceil(self.mid) + 1

    def opposite_best(self) -> int | None:
        return self.best_ask if self.side == "bid" else self.best_bid

    def behind(self, ticks: int) -> int:
        """Price ``ticks`` behind own best (less aggressive)."""
        return self.own_best() - ticks if self.side == "bid" else self.own_best() + ticks


class Strategy:
    """A baseline or learned policy. ``rng`` is the executor's own stream; ``reset`` is called per episode."""

    name = "strategy"

    def reset(self, state: ExecState, rng) -> None:
        pass

    def decide(self, state: ExecState, rng) -> Decision:
        raise NotImplementedError


@dataclass
class ExecutionResult:
    side: str
    quantity: int
    arrival_mid: float
    start: int
    executed: int = 0
    forced: int = 0                       # lots pushed through by market order at the horizon
    notional: int = 0
    end: int | None = None                # step the order was done
    aggressive_executed: int = 0
    orders_sent: int = 0
    incomplete: int = 0                   # lots that could not be executed even by force
    fills: list = field(default_factory=list)   # (step, avg price, quantity, aggressive)

    @property
    def avg_price(self) -> float | None:
        return self.notional / self.executed if self.executed else None

    @property
    def shortfall_bps(self) -> float | None:
        """Average fill price against the arrival mid, positive = cost; forced completion included."""
        if not self.executed:
            return None
        sign = 1 if self.side == "bid" else -1
        return sign * (self.avg_price - self.arrival_mid) / self.arrival_mid * 1e4

    @property
    def completion(self) -> float:
        """Share of the order filled before the horizon (forced completion excluded)."""
        return (self.executed - self.forced) / self.quantity if self.quantity else 1.0


class ExecutorAgent(Agent):
    """Carries ``strategy`` through ``task`` from step ``start`` with the task's hard limits."""

    def __init__(self, *args, task: ExecutionTask, strategy: Strategy, start: int, expected_volume: float):
        super().__init__(*args)
        self.task, self.strategy, self.start = task, strategy, start
        self.expected_volume = expected_volume
        self.quantity = max(1, round(task.size_fraction * expected_volume * task.horizon))
        self.result: ExecutionResult | None = None
        self.done = False
        self.public_volume_since_start = 0
        self.own_aggressive = 0
        self.orders_sent = 0
        self._own_window: list[int] = []   # own aggressive lots per step
        self._seen_shares = 0

    # ----- accounting ------------------------------------------------------ #

    def _executed(self) -> int:
        delta = self.account.shares - self._start_shares
        return delta if self.task.side == "bid" else -delta

    def _record(self, aggressive: bool) -> None:
        """Turn the account change since the last call into a fill record."""
        now = self._executed()
        qty = now - self._seen_shares
        if qty > 0:
            notional_now = abs(self.account.cash - self._start_cash)
            price = (notional_now - self._seen_notional) / qty
            self.result.fills.append((self.market.step, price, qty, aggressive))
            if aggressive:
                self.own_aggressive += qty
                self._own_window[-1] += qty
            self._seen_shares, self._seen_notional = now, notional_now

    def state(self) -> ExecState:
        m, ps, task = self.market, self.market.public, self.task
        elapsed = m.step - self.start
        decisions_left = max(1, math.ceil((task.horizon - elapsed) / task.decision_interval))
        resting = m.book.get_client_orders(self.cid)
        window = ps.flow_window
        window_flow = ps.buy_flow + ps.sell_flow
        own_window = sum(self._own_window[-window:])
        executed = self._executed()
        return ExecState(
            step=m.step, elapsed=elapsed, horizon=task.horizon, decision_interval=task.decision_interval,
            decisions_left=decisions_left, side=task.side, quantity=self.quantity,
            remaining=self.quantity - executed, executed=executed, aggressive_executed=self.own_aggressive,
            arrival_mid=self.result.arrival_mid, mid=ps.mid, best_bid=ps.best_bid, best_ask=ps.best_ask,
            spread=ps.spread, imbalance=ps.imbalance, public_volume_since_start=self.public_volume_since_start,
            own_resting=sum(o["quantity"] for o in resting), own_resting_orders=resting,
            expected_volume_per_step=self.expected_volume,
            own_flow_share=min(1.0, own_window / window_flow) if window_flow else 0.0, public=ps,
        )

    # ----- stepping -------------------------------------------------------- #

    def act(self) -> None:
        m, task = self.market, self.task
        if self.done or m.step < self.start:
            return
        ps = m.public
        if self.result is None:
            self._start_cash, self._start_shares = self.account.cash, self.account.shares
            self._seen_shares = self._seen_notional = 0
            self.result = ExecutionResult(task.side, self.quantity, ps.mid, m.step)
            self._own_window.append(0)
            self.strategy.reset(self.state(), self.rng)
        else:
            self.public_volume_since_start += ps.step_volume
            self._own_window.append(0)
            self._record(aggressive=False)  # passive fills since the last step
        elapsed = m.step - self.start
        if elapsed >= task.horizon:
            self._force()
        elif self._executed() >= self.quantity:
            self._finish()
        elif elapsed % task.decision_interval == 0:
            self._apply(self.strategy.decide(self.state(), self.rng))

    def _apply(self, d: Decision) -> None:
        m, cid, task = self.market, self.cid, self.task
        if d.cancel and m.book.client_order_ids(cid):
            m.cancel_all(cid)
        remaining = self.quantity - self._executed()
        qty = min(int(d.market_qty), remaining)
        if task.max_participation is not None:
            qty = min(qty, max(1, round(task.max_participation * self.expected_volume * task.decision_interval)))
        if qty >= 1:
            m.market(cid, task.side, qty)
            self.orders_sent += 1
            self._record(aggressive=True)
        resting = sum(o["quantity"] for o in m.book.get_client_orders(cid))
        lqty = min(int(d.limit_qty), self.quantity - self._executed() - resting)
        if lqty >= 1 and len(m.book.client_order_ids(cid)) < task.max_resting_orders:
            state = self.state()
            price = d.limit_price if d.limit_price is not None else state.own_best()
            opposite = state.opposite_best()
            if opposite is not None:  # a marketable limit may not go further through the best than allowed
                limit = opposite + task.max_through_ticks if task.side == "bid" else opposite - task.max_through_ticks
                price = min(price, limit) if task.side == "bid" else max(price, limit)
            report = m.limit(cid, task.side, int(price), lqty)
            if report is not None:
                self.orders_sent += 1
                if report["filled_quantity"]:
                    self._record(aggressive=True)

    def _force(self) -> None:
        """At the horizon: cancel what rests and push the remainder through by market order."""
        m, cid = self.market, self.cid
        if m.book.client_order_ids(cid):
            m.cancel_all(cid)
        remaining = self.quantity - self._executed()
        if remaining > 0:
            before = self._executed()
            m.market(cid, self.task.side, remaining)
            self.orders_sent += 1
            self._record(aggressive=True)
            self.result.forced += self._executed() - before
        if self._executed() >= self.quantity or m.step - self.start >= self.task.horizon + self.task.grace:
            self._finish()

    def _finish(self) -> None:
        m, r = self.market, self.result
        if m.book.client_order_ids(self.cid):
            m.cancel_all(self.cid)
        self._record(aggressive=False)
        r.executed = self._executed()
        r.notional = abs(self.account.cash - self._start_cash)
        r.end = m.step
        r.aggressive_executed = self.own_aggressive
        r.orders_sent = self.orders_sent
        r.incomplete = self.quantity - r.executed
        self.done = True


def expected_volume_from_tape(tape: list, steps: int, fallback: float = 5.0) -> float:
    """Mean traded lots per step over the last ``steps`` steps of an analysis tape."""
    if not tape or steps <= 0:
        return fallback
    last = tape[-1][0]
    return sum(t[2] for t in tape if t[0] > last - steps) / steps
