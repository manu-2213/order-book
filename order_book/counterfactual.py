"""Two worlds in lockstep: Arm 0 and a treated world, with four reaction channels on independent switches.

Every run steps a *reference* world (Arm 0: no executor) next to a *treated*
world (the executor is there). Each channel decides whether the background's
reaction to the executor exists in the treated world:

    I  inventory   makers skew their reservation on the inventory the executor pushed onto them
    P  price       the background lives on the true book: its public state is the treated one
    D  detection   the makers' flow detector sees the executor's flow
    A  predation   the order anticipator trades, on a detector that sees the executor's flow

All four off is the ghost arm (``order_book.ghost``). All four on is the full
A4 market. How a channel is realised depends on P, since P decides where the
background's orders come from:

    P on   The treated world is a clone of the reference, its agents act there on
           what they see there. I off points the makers' skew at their Arm-0
           accounts. D off leaves the executor out of the flow their detector reads.
    P off  The background acts in the reference and its order flow is replayed
           on the copy. I on and D on shade each replayed maker quote by the
           difference between the copy's inventory or signal and Arm 0's, using
           the maker's own formula (``MarketMaker.shade``), so no order is created
           or retimed and the shade is zero when nothing has diverged.

A runs the same way in both: a live anticipator in the treated world. The
population always contains it, dormant when off, so all arms share one scheduler
permutation and the reference never contains an active one.

Each pair of agents acts in the same scheduled position, treated first, so a
treated agent reads the reference exactly as its twin is about to. With no
executor, and A off, the two worlds are therefore identical in every
configuration, which is the unit test that keeps the channels honest.

The natural ladder of the research plan is the chain of nested arms ``ARMS``;
its increments depend on the order the channels join, so ``ALL_CHANNELS`` spans
the 2^4 factorial that Shapley values need (``papers/decomposition``).
"""

import copy
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import product

import numpy as np

from order_book.agents import MarketMaker, OrderAnticipator
from order_book.config import SimConfig
from order_book.execution import Calibration, ExecutionTask, ExecutorAgent, expected_volume_from_tape, make_strategy
from order_book.ghost import GhostConfig, GhostMarket
from order_book.presets import adversary
from order_book.simulator import Simulator

EXECUTOR_ENDOWMENT = 50_000          # shares; cash = shares x mid, so any task size is funded


@dataclass(frozen=True)
class Channels:
    """Which reactions exist in the treated world.

    The price channel holds several reactions at once, so three controls (only with ``price``) take
    one out by pointing that group at Arm 0's public state while everything else stays live:
    ``chartists`` read the treated momentum (price-chasing), ``makers_follow`` centre their quotes on
    the treated mid and read its volatility, ``fundamentalists`` compare their belief with the treated
    mid (leaning against the move). ``blind_predator`` (only without ``predation``): the anticipator
    trades but cannot see the executor, measuring what its activity on natural flow does by itself.
    """

    inventory: bool = False
    price: bool = False
    detection: bool = False
    predation: bool = False
    chartists: bool = True
    makers_follow: bool = True
    fundamentalists: bool = True
    blind_predator: bool = False

    @property
    def key(self) -> str:
        base = "".join(c if on else "-" for c, on in zip("IPDA", (self.inventory, self.price, self.detection, self.predation)))
        return (base + ("" if self.chartists else "c") + ("" if self.makers_follow else "m")
                + ("" if self.fundamentalists else "f") + ("b" if self.blind_predator else ""))

    @property
    def members(self) -> frozenset[str]:
        return frozenset(self.key[:4].replace("-", ""))


CHANNEL_NAMES = {"I": "inventory", "P": "price", "D": "detection", "A": "predation"}
ALL_CHANNELS = tuple(Channels(i, p, d, a) for i, p, d, a in product((False, True), repeat=4))
GHOST = Channels()
FULL = Channels(True, True, True, True)
# The plan's ladder, in the plan's order, by intervention. Each arm adds one thing to the one before.
ARMS = {
    1: GHOST,                                  # copy of the book; nobody sees anything
    2: Channels(price=True),                   # true book, reservation on Arm-0 inventory, detector blind
    3: Channels(True, True),                   # + the makers' real inventory
    4: Channels(True, True, True),             # + the detector sees the executor (A1)
    5: FULL,                                   # + the anticipator (A4)
}
# The same chain with inventory joining before price: the order the plan's "increment isolates" column names.
PLAN_LABEL_ORDER = {1: GHOST, 2: Channels(inventory=True), 3: Channels(True, True), 4: ARMS[4], 5: FULL}
# Controls outside the factorial: the price arm with one group blind; A1 plus a blind anticipator.
CHARTISTS_BLIND = Channels(price=True, chartists=False)
MAKERS_BLIND = Channels(price=True, makers_follow=False)
FUNDAMENTALISTS_BLIND = Channels(price=True, fundamentalists=False)
PREDATOR_CONTROL = Channels(True, True, True, False, blind_predator=True)


class OrderLog:
    """A ``Market.mirror`` that records the order flow (step, client, side, quantity), for coupling diagnostics."""

    def __init__(self, market):
        self.market, self.orders = market, []
        market.mirror = self

    def on_order(self, report) -> None:
        self.orders.append((self.market.step, report["client_id"], report["side"], report["requested_quantity"]))

    def on_cancel(self, cid, order_id) -> None:
        pass


class PairedRun:
    """A burnt-in reference world and one treated world stepping in lockstep under ``channels``.

    ``market`` is the market the executor trades on and its ``public`` what it observes.
    """

    def __init__(self, blob: bytes, channels: Channels = FULL, ghost: GhostConfig | None = None):
        self.channels = channels
        self.reference = reference = Simulator.restore(blob)
        n = len(reference.agents)
        self.executor = None
        self.treated = None
        self.ghost = None
        if channels.price:
            self.treated = treated = Simulator.restore(blob)
            self.market = treated.market
            self.flow = treated.detectors["flow"]
            self.flow.include_hidden = channels.detection
            if not channels.inventory:
                for twin, own in zip(reference.agents, treated.agents):
                    if isinstance(own, MarketMaker):
                        own.skew_account = twin.account
            arm0_public = reference.market.public
            if not channels.makers_follow:
                for own in treated.agents:
                    if isinstance(own, MarketMaker):
                        own.public_view = arm0_public
            if not channels.chartists:
                treated.speculators.momentum_public = arm0_public
            if not channels.fundamentalists:
                treated.speculators.belief_public = arm0_public
            self._treated_agents = treated.agents
        else:
            self.ghost = GhostMarket(reference.market, ghost)
            self.market = self.ghost
            self.flow = copy.copy(reference.detectors["flow"])
            self.flow.include_hidden = True
            if channels.inventory or channels.detection:
                self.ghost.quote_shade = self._shader()
            self._treated_agents = [None] * n
        self.detectors = [self.flow]
        self._wire_anticipator()

    def _wire_anticipator(self) -> None:
        channels, reference = self.channels, self.reference
        if not channels.predation and not channels.blind_predator:
            return
        for i, twin in enumerate(reference.agents):
            if isinstance(twin, OrderAnticipator):
                if self.treated is not None:
                    agent = self.treated.agents[i]
                else:  # a live trader in the copy, on its own shadow account
                    agent = copy.copy(twin)
                    agent.market, agent.account = self.ghost, self.ghost.accounts[twin.cid]
                    self._treated_agents[i] = agent
                predator_flow = copy.copy(self.flow)
                predator_flow.include_hidden = channels.predation
                agent.detector = predator_flow
                agent.active = True
                self.detectors.append(predator_flow)

    def _shader(self):
        makers = {a.cid: a for a in self.reference.agents if isinstance(a, MarketMaker)}
        ghost, flow, channels = self.ghost, self.flow, self.channels

        def shade(cid, side, price, quantity):
            maker = makers.get(cid)
            if maker is None:
                return price, quantity
            account = ghost.accounts[cid] if channels.inventory else maker.account
            signal = 0.0
            if maker.detector is not None:
                signal = flow.signal if channels.detection else maker.detector.signal
            return maker.shade(side, price, quantity, account.shares - maker.target, signal)

        return shade

    def add_executor(self, task: ExecutionTask, strategy, volume: float, endowment: int = EXECUTOR_ENDOWMENT):
        """Add the executor at the current step; it is hidden, and detectors that may see it read the full flow."""
        cash = endowment * self.reference.config.book.mid
        common = dict(task=task, strategy=strategy, start=self.market.step, expected_volume=volume)
        if self.treated is not None:
            self.executor = self.treated.add_external(ExecutorAgent, cash, endowment, **common)
        else:
            cid = max(self.ghost.accounts) + 1
            self.executor = ExecutorAgent(cid, self.ghost, self.reference.new_stream("Ext1"), cash, endowment, **common)
        self.market.hide(self.executor.cid)
        return self.executor

    def step(self) -> None:
        reference, treated, executor = self.reference, self.treated, self.executor
        reference.fundamental.advance()
        if treated is not None:
            treated.fundamental.advance()
        if executor is not None:
            executor.act()
        order = reference.schedule()
        if treated is not None:
            treated.schedule()  # same draws; keeps a stateful scheduler stream aligned
        own, twins = self._treated_agents, reference.agents
        for i in order:
            agent = own[i]
            if agent is not None:
                agent.act()
            twins[i].act()
        reference.market.end_step()
        self.market.end_step()
        for detector in reference.detectors.values():
            detector.update(reference.market.public)
        for detector in self.detectors:
            detector.update(self.market.public)


# ----- one run ----------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairedSpec:
    seed: int
    strategy: str
    size_fraction: float
    channels: Channels = FULL
    side: str | None = None         # None: "bid" on even seeds, "ask" on odd seeds
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    after: int = 500                # steps recorded after the horizon (reversion)
    threshold: str = "medium"       # flow-detector threshold of the A1/A4 market
    detecting: int | None = None    # detecting makers; None: the preset's min(2, n)
    base_config: SimConfig | None = None
    regime: str = "default"         # label; the market itself is in base_config
    rng: str = "counter"            # "counter" or "stateful"
    ghost: GhostConfig | None = None
    record_path: bool = False

    @property
    def resolved_side(self) -> str:
        return self.side or ("bid" if self.seed % 2 == 0 else "ask")

    @property
    def window(self) -> int:
        return self.horizon + self.after

    def config(self) -> SimConfig:
        return replace(adversary("A4", self.threshold, self.base_config, self.detecting), rng=self.rng)

    @property
    def key(self) -> tuple:
        """What identifies this run in a result log; ``PairedResult.key`` is the same tuple."""
        return self.seed, self.strategy, self.size_fraction, self.horizon, self.channels.key, self.regime, self.rng


@dataclass
class PairedResult:
    seed: int
    strategy: str
    size_fraction: float
    horizon: int
    channels: str                   # ``Channels.key``
    regime: str
    rng: str
    side: str
    quantity: int
    executed: int
    forced: int
    incomplete: int
    completion: float
    shortfall_bps: float | None     # the executor's implementation shortfall in this world
    drift_bps: float                # what the same fills would have cost at Arm 0's mid: natural drift, no footprint
    aggressive_share: float
    orders_sent: int
    end: int                        # step (from the arrival) the order was done
    volume: float                   # expected market volume per step, from the burn-in (sizes the order)
    q_over_v: float                 # realised: executed / Arm 0's traded volume over the horizon
    impact_mean: float              # signed treated-minus-Arm-0 mid, mean while executing, ticks
    impact_end: float               # the same at completion
    impact_peak: float              # its maximum up to completion
    impact_residual: float          # its value ``after`` steps past completion (or at the end of the window)
    quote_lift: float               # mean signed move of the opposite best against Arm 0 while executing, ticks
    spread_change: float            # mean spread minus Arm 0's while executing, ticks
    flow_firing: float              # mean |signal| of the maker-facing detector while executing
    ghost_stats: dict | None
    impact_path: list | None = None
    seconds: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(record: dict) -> "PairedResult":
        return PairedResult(**record)

    @property
    def key(self) -> tuple:
        return self.seed, self.strategy, self.size_fraction, self.horizon, self.channels, self.regime, self.rng


@lru_cache(maxsize=16)
def burned_in(seed: int, cfg: SimConfig, burn_in: int) -> tuple[bytes, float]:
    """(snapshot, expected volume per step) of Arm 0 after the burn-in. The tape is only needed for the volume."""
    sim = Simulator(seed, cfg, analysis=True)
    for agent in sim.agents:
        if isinstance(agent, OrderAnticipator):
            agent.active = False
    sim.run(burn_in)
    volume = expected_volume_from_tape(sim.market.tape, burn_in)
    sim.market.tape = None
    sim.market.analysis = False
    sim.book.keep_trades = False
    sim.book._trades = []
    sim.fundamental.history = None
    return sim.snapshot(), volume


def run_paired(spec: PairedSpec, calibration: Calibration | None = None) -> PairedResult:
    t0 = time.perf_counter()
    blob, volume = burned_in(spec.seed, spec.config(), spec.burn_in)
    if spec.strategy.startswith("rl@"):
        import torch
        torch.set_num_threads(1)  # one policy evaluation at a time; workers are the unit of parallelism
    run = PairedRun(blob, spec.channels, spec.ghost)
    task = ExecutionTask(side=spec.resolved_side, size_fraction=spec.size_fraction, horizon=spec.horizon,
                         decision_interval=spec.decision_interval)
    executor = run.add_executor(task, make_strategy(spec.strategy, calibration), volume)
    reference_public, public = run.reference.market.public, run.market.public
    window = spec.window
    sign = 1 if task.side == "bid" else -1
    path = np.empty(window)
    arm0_mid = np.empty(window + 1)
    arm0_mid[0] = reference_public.mid
    lift = spread = firing = 0.0
    active = arm0_traded = 0
    for t in range(window):
        run.step()
        path[t] = sign * (public.mid - reference_public.mid)
        arm0_mid[t + 1] = reference_public.mid
        if t < spec.horizon:
            arm0_traded += reference_public.step_volume
        if not executor.done:
            opposite, arm0_opposite = public.best_ask, reference_public.best_ask
            if sign < 0:
                opposite, arm0_opposite = public.best_bid, reference_public.best_bid
            if opposite is not None and arm0_opposite is not None:
                lift += sign * (opposite - arm0_opposite)
            spread += public.spread - reference_public.spread
            firing += abs(run.flow.signal)
            active += 1
    if not executor.done:
        executor._finish()
    r = executor.result
    end = (r.end - executor.start) if r.end is not None else spec.horizon
    last = min(end, window - 1)
    drift = 0.0
    if r.executed:
        start = executor.start
        cost = sum(qty * (arm0_mid[min(step - start, window)] - r.arrival_mid) for step, _, qty, _ in r.fills)
        drift = sign * cost / (r.executed * r.arrival_mid) * 1e4
    active = max(active, 1)
    return PairedResult(
        spec.seed, spec.strategy, spec.size_fraction, spec.horizon, spec.channels.key, spec.regime, spec.rng, task.side,
        r.quantity, r.executed, r.forced, r.incomplete, r.completion, r.shortfall_bps, float(drift),
        r.aggressive_executed / r.executed if r.executed else 0.0, r.orders_sent, end, volume,
        r.executed / max(arm0_traded, 1),
        float(path[:last + 1].mean()), float(path[last]), float(path[:last + 1].max()),
        float(path[min(last + spec.after, window - 1)]),
        lift / active, spread / active, firing / active,
        dict(run.ghost.stats) if run.ghost is not None else None,
        path.tolist() if spec.record_path else None, time.perf_counter() - t0,
    )


def flow_coupling(spec: PairedSpec, calibration: Calibration | None = None, buckets: int = 5) -> list[float]:
    """How much of the background's order flow the two worlds still share, per stretch of the window.

    Jaccard overlap of the multiset of (step, client, side, size) orders, in equal time buckets, in a
    true-book arm. A stateful stream that has shifted loses it for good; a keyed one only where the
    agent genuinely decided otherwise.
    """
    from collections import Counter

    if not spec.channels.price:
        raise ValueError("flow_coupling compares two live worlds, so it needs the price channel on")
    blob, volume = burned_in(spec.seed, spec.config(), spec.burn_in)
    run = PairedRun(blob, spec.channels, spec.ghost)
    task = ExecutionTask(side=spec.resolved_side, size_fraction=spec.size_fraction, horizon=spec.horizon,
                         decision_interval=spec.decision_interval)
    executor = run.add_executor(task, make_strategy(spec.strategy, calibration), volume)
    start = run.market.step
    logs = OrderLog(run.reference.market), OrderLog(run.market)
    for _ in range(spec.window):
        run.step()
    overlap = []
    for b in range(buckets):
        lo, hi = start + b * spec.window // buckets, start + (b + 1) * spec.window // buckets
        a, c = (Counter(o for o in log.orders if lo <= o[0] < hi and o[1] != executor.cid) for log in logs)
        union = sum((a | c).values())
        overlap.append(sum((a & c).values()) / union if union else 1.0)
    return overlap
