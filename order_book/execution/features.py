"""Observation and action encoding for the learned executor (plan section 5).

The feature vector is built from the executor's own ``ExecState`` only, so the
policy sees public market data and its own position -- never the detectors it
is hiding from. The same encoder is used while training (``ExecutionEnv``) and
while evaluating the trained policy as an ordinary ``Strategy``
(``PolicyStrategy``), so there is a single definition of the state.

The action is a discrete pair ``(size multiplier, placement)``: how large the
next child is relative to the remaining-over-decisions-left pace, and whether
it is a market order, a passive limit at own best, a passive limit one tick
behind, or a limit one tick inside the spread. A multiplier of zero is "hold",
whatever the placement. Actions and features only ever grow by appending, so
a saved net keeps its meaning: it reads the feature prefix it was trained on
and its own action count.
"""

import math
from dataclasses import dataclass

import numpy as np

from order_book.execution.task import Decision, ExecState

# (size multiplier, placement) grid. Placement: 0 market, 1 passive at own best, 2 passive one tick behind; then, appended
# later so that the first 15 indices keep their meaning, a limit one tick inside the spread at each multiplier.
SIZE_MULTS = (0.0, 0.5, 1.0, 1.5, 2.5)
PLACEMENTS = ("market", "passive_best", "passive_behind")
ACTIONS: tuple[tuple[float, str], ...] = tuple(
    (mult, place) for mult in SIZE_MULTS for place in PLACEMENTS
) + tuple((mult, "improve") for mult in SIZE_MULTS)
N_ACTIONS = len(ACTIONS)
N_FEATURES = 19  # append new features at the end: a saved net reads the prefix it was trained on
MAX_BPS = 20.0  # scale for the running paper-loss feature


def features(s: ExecState) -> np.ndarray:
    """A fixed-length, roughly unit-scaled view of one decision point (public data + own state)."""
    horizon_decisions = max(1, s.horizon / s.decision_interval)
    time_frac = s.elapsed / s.horizon if s.horizon else 1.0
    remain_frac = s.remaining / s.quantity if s.quantity else 0.0
    schedule_gap = (s.executed - s.quantity * time_frac) / s.quantity if s.quantity else 0.0
    paper_bps = s.sign * (s.mid - s.arrival_mid) / s.arrival_mid * 1e4 if s.arrival_mid else 0.0
    mom = s.public.momentum
    vol = math.sqrt(max(s.public.var_fast, 0.0))
    vec = np.array([
        time_frac,
        remain_frac,
        np.clip(schedule_gap, -1.0, 1.0),
        s.decisions_left / horizon_decisions,
        min(s.spread / 5.0, 2.0),
        s.sign * s.imbalance,
        s.own_flow_share,
        np.clip(s.sign * mom[0] * 1e2, -3.0, 3.0),
        np.clip(s.sign * mom[-1] * 1e2, -3.0, 3.0),
        np.clip(paper_bps / MAX_BPS, -3.0, 3.0),
        min(s.own_resting / s.quantity, 2.0) if s.quantity else 0.0,
        min(vol / 2.0, 3.0),
        np.clip(s.public.attention - 1.0, -1.0, 2.0),
        min(s.quantity / max(s.expected_volume_per_step * s.horizon, 1.0), 1.0),  # nominal participation: the order's size for this market
        min(s.own_best_depth / 50.0, 3.0),                   # 14-16: the book at the touch and the agent's place in it
        min(s.opposite_best_depth / 50.0, 3.0),
        min(s.own_queue_ahead / 50.0, 3.0),
        np.clip(s.sign * mom[0] / 0.5, -3.0, 3.0),           # 17-18: momentum again, scaled to its spread (7-8 sit at their clip)
        np.clip(s.sign * mom[-1] / 1.0, -3.0, 3.0),
    ], dtype=np.float32)
    return vec


def decision(action: int, s: ExecState) -> Decision:
    """Turn a discrete action index into a concrete ``Decision`` for the current state."""
    mult, place = ACTIONS[int(action)]
    base = s.remaining / s.decisions_left  # at least 1 by construction
    qty = max(0, round(mult * base))
    if qty <= 0:
        return Decision()
    if place == "market":
        return Decision(cancel=True, market_qty=qty)
    price = None if place == "passive_best" else s.inside(1) if place == "improve" else s.behind(1)
    return Decision(cancel=True, limit_qty=qty, limit_price=price)


@dataclass
class Transition:
    """One recorded decision: features, chosen action, its log-prob and value, and the reward that follows."""

    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
