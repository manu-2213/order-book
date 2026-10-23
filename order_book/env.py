"""Gymnasium wrapper: one RL trader inside the background market.

The agent decides every ``decision_interval`` simulator steps. Episodes start
from a pool of saved post-burn-in simulator states (pickled ``Simulator``
objects, several seeds), so burn-in is paid once, not per episode. The env is
picklable and seedable, so it runs under ``gymnasium.vector.AsyncVectorEnv``.

    python -m scripts.burn_in --seeds 8 --burn-in 5000      # writes outputs/states.pkl
"""

import math
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np

from order_book.config import SimConfig
from order_book.simulator import Simulator

STATES_PATH = Path(__file__).resolve().parents[1] / "outputs" / "states.pkl"


def burn_in(seeds, steps: int = 5_000, config: SimConfig | None = None) -> list[bytes]:
    """Training-mode simulators run for ``steps`` and snapshotted, one per seed."""
    states = []
    for seed in seeds:
        sim = Simulator(int(seed), config, analysis=False)
        sim.run(steps)
        states.append(sim.snapshot())
    return states


class OrderBookEnv(gym.Env):
    """Discrete actions, mark-to-market reward in ticks.

    Actions: 0 hold, 1 market buy, 2 market sell, 3 join best bid, 4 join best
    ask, 5 cancel all (orders are ``order_size`` lots). Observation: spread,
    the three public momentum indicators, fast volatility, flow imbalance and
    signal, attention, mid minus slow MA, own position, own resting bid/ask
    lots, free cash in lots, then ``depth`` bid and ask volumes from the best.
    """

    metadata = {"render_modes": []}
    ACTIONS = ("hold", "market_buy", "market_sell", "join_bid", "join_ask", "cancel_all")

    def __init__(self, states: list[bytes] | str | Path | None = None, decision_interval: int = 10,
                 horizon: int = 10_000, cash: int = 1_000_000, shares: int = 100, order_size: int = 5,
                 depth: int = 5):
        if states is None or isinstance(states, (str, Path)):
            with open(STATES_PATH if states is None else states, "rb") as f:
                states = pickle.load(f)
        if not states or min(decision_interval, horizon, order_size) < 1 or depth < 0:
            raise ValueError("need at least one state and positive interval, horizon and order size")
        self.states, self.decision_interval, self.horizon = list(states), decision_interval, horizon
        self.cash, self.shares, self.order_size, self.depth = cash, shares, order_size, depth
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (13 + 2 * depth,), np.float32)
        self.sim: Simulator | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = Simulator.restore(self.states[self.np_random.integers(len(self.states))])
        self.sim.reseed(int(self.np_random.integers(1 << 31)))
        self.cid = self.sim.join(self.cash, self.shares)
        self.account = self.sim.market.accounts[self.cid]
        self.decisions = 0
        return self._observation(), {}

    def step(self, action: int):
        m, cid, size = self.sim.market, self.cid, self.order_size
        book = m.book
        if action == 1:
            m.market(cid, "bid", size)
        elif action == 2:
            m.market(cid, "ask", size)
        elif action == 3:
            m.limit(cid, "bid", book.best_bid if book.best_bid is not None else math.floor(m.public.mid) - 1, size)
        elif action == 4:
            m.limit(cid, "ask", book.best_ask if book.best_ask is not None else math.ceil(m.public.mid) + 1, size)
        elif action == 5:
            m.cancel_all(cid)
        before = m.wealth(cid)
        self.sim.run(self.decision_interval)
        self.decisions += 1
        reward = m.wealth(cid) - before
        return self._observation(), reward, False, self.decisions >= self.horizon, {"wealth": m.wealth(cid)}

    def _observation(self) -> np.ndarray:
        m, ps, book = self.sim.market, self.sim.market.public, self.sim.book
        resting = {"bid": 0, "ask": 0}
        for order in book.get_client_orders(self.cid):
            resting[order["side"]] += order["quantity"]
        arrays = book.get_book_arrays(self.depth)
        features = [
            ps.spread, *ps.momentum, math.sqrt(ps.var_fast), ps.imbalance, self.sim.detectors["flow"].signal, ps.attention,
            ps.mid - ps.ma_slow, self.account.shares - self.shares, resting["bid"], resting["ask"],
            self.account.free_cash / ps.mid,
        ]
        return np.concatenate([features, arrays["bid_volumes"], arrays["ask_volumes"]]).astype(np.float32)

