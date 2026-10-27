"""The execution MDP for Phase 3: a learned executor inside the agent-based book (plan section 5).

The environment is the Phase-2 measurement machinery run for learning. One
``ExecutorAgent`` carries a ``PolicyStrategy`` through an ``ExecutionTask`` in a
market with detecting market makers; the environment rolls out a whole episode,
lets the policy decide every ``decision_interval`` steps, and turns realised
fills into a telescoping implementation-shortfall reward (bps, negated so that
higher is cheaper). The reward summed over an episode equals minus the
quantity-weighted shortfall, so a return-maximising policy minimises cost.

The default market is the Phase-2 "default strength, low-cover" regime that E7b
identified as the one where TWAP leaks measurably while the book stays stable:
a liquid book with a single institution for cover and two detecting makers.
Pass any ``SimConfig`` to study another regime.
"""

from dataclasses import dataclass, field

import numpy as np

from order_book.config import SimConfig
from order_book.execution.features import Transition
from order_book.execution.policy import PolicyStrategy
from order_book.execution.task import ExecutionTask, ExecutorAgent, expected_volume_from_tape
from order_book.presets import adversary, regime
from order_book.simulator import Simulator

EXECUTOR_ENDOWMENT = 50_000          # shares; cash = shares x mid, so any task size is funded


def phase3_config(institutions: int = 1, liquidity: float = 1.0, threshold: str = "medium",
                  detecting: int = 2, level: str = "A1") -> SimConfig:
    """The E7b 'default strength, low-cover' market: liquid book, little cover, detecting makers."""
    return adversary(level, threshold, base=regime(liquidity, institutions), detecting=detecting)


@dataclass
class EpisodeInfo:
    """Summary of one rolled-out episode (what the trainer and tests look at)."""

    shortfall_bps: float | None
    leakage_proxy: float
    completion: float
    executed: int
    forced: int
    decisions: int
    ret: float                      # sum of rewards (= -quantity-weighted shortfall)


@dataclass
class ExecutionEnv:
    """Rolls out full execution episodes for on-policy training and evaluation.

    ``rollout`` runs one episode with a given strategy; when the strategy records
    transitions (``PolicyStrategy`` with a ``recorder``) the per-decision rewards
    are filled in and the transition list is returned for PPO.
    """

    config: SimConfig = field(default_factory=phase3_config)
    side: str | None = None          # None: "bid" on even seeds, "ask" on odd seeds
    size_fraction: float = 0.20
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    incomplete_penalty: float = 5.0  # bps per unfilled unit-fraction, charged on the last transition

    def resolved_side(self, seed: int) -> str:
        return self.side if self.side is not None else ("bid" if seed % 2 == 0 else "ask")

    def rollout(self, strategy, seed: int) -> tuple[list[Transition], EpisodeInfo]:
        """Run one episode; returns (transitions, info). Transitions are empty unless the strategy records."""
        if isinstance(strategy, PolicyStrategy):
            strategy.recorder = []
        cfg = self.config
        sim = Simulator(seed, cfg, analysis=True)
        task = ExecutionTask(side=self.resolved_side(seed), size_fraction=self.size_fraction,
                             horizon=self.horizon, decision_interval=self.decision_interval)
        executor = sim.add_external(ExecutorAgent, EXECUTOR_ENDOWMENT * cfg.book.mid, EXECUTOR_ENDOWMENT,
                                    task=task, strategy=strategy, start=self.burn_in, expected_volume=1.0)
        sim.run(self.burn_in)
        volume = expected_volume_from_tape(sim.market.tape, self.burn_in)
        executor.expected_volume = volume
        executor.quantity = max(1, round(self.size_fraction * volume * self.horizon))
        quantity, arrival = executor.quantity, None
        transitions = strategy.recorder if isinstance(strategy, PolicyStrategy) else []
        seen_fills, window = 0, self.horizon + task.grace + 1
        for _ in range(window):
            sim.step()
            if executor.result is None:
                continue
            if arrival is None:
                arrival = executor.result.arrival_mid
            fills = executor.result.fills
            while seen_fills < len(fills) and transitions:
                _, price, qty, _aggr = fills[seen_fills]
                cost_bps = task_sign(task.side) * (price - arrival) / arrival * 1e4
                transitions[-1].reward += -(cost_bps) * (qty / quantity)
                seen_fills += 1
            if executor.done:
                break
        r = executor.result
        if transitions:
            incomplete_frac = (r.incomplete / quantity) if r.incomplete else 0.0
            transitions[-1].reward -= self.incomplete_penalty * incomplete_frac
            transitions[-1].done = True
        ret = sum(t.reward for t in transitions)
        info = EpisodeInfo(
            shortfall_bps=r.shortfall_bps, leakage_proxy=r.aggressive_executed / r.executed if r.executed else 0.0,
            completion=r.completion, executed=r.executed, forced=r.forced, decisions=len(transitions), ret=ret,
        )
        return transitions, info


def task_sign(side: str) -> int:
    return 1 if side == "bid" else -1
