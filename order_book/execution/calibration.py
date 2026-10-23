"""Fit the model-based baselines' parameters from A0 runs (plan section 7, "Calibration and notes").

    sigma               std of one-step mid changes on the background market
    temporary_impact    slope of mean child slippage (fill price - mid before the child, in the
                        order's direction) on the trading rate, from TWAP-market runs of several sizes
    permanent_impact    paired impact at completion per lot (``order_book.impact``)
    resilience          exponential decay rate fitted to the paired impact path after completion
    depth               mean resting lots within 5 ticks of the best on one side
    fill_prob           share of 1-lot probes posted d ticks behind own best that fill within N steps

Everything is measured under the A0 preset (no detection) on training seeds.
"""

import json
import math
from dataclasses import asdict, replace

import numpy as np

from order_book.agents.base import Agent
from order_book.config import SimConfig
from order_book.execution.strategies import Calibration, TwapMarket
from order_book.execution.task import ExecutionTask, ExecutorAgent, expected_volume_from_tape
from order_book.presets import adversary
from order_book.simulator import Simulator


class FillProber(Agent):
    """Posts a 1-lot limit ``distance`` ticks behind its own best every ``interval`` steps on alternating
    sides, cancels it at the next decision, and counts fills. Public data and own orders only."""

    def __init__(self, *args, distances=(0, 1, 2, 3), interval: int = 5):
        super().__init__(*args)
        self.distances, self.interval = distances, interval
        self.posted = {d: 0 for d in distances}
        self.filled = {d: 0 for d in distances}
        self._live: list[tuple[int, int, int]] = []   # (order_id, distance, shares_before)
        self._k = 0

    def act(self) -> None:
        m, book = self.market, self.market.book
        if m.step % self.interval:
            return
        for order_id, d, shares_before in self._live:
            if book.get_order(order_id) is None:
                self.filled[d] += 1
            else:
                m.cancel(self.cid, order_id)
        self._live = []
        side = "bid" if self._k % 2 == 0 else "ask"
        self._k += 1
        best = book.best_bid if side == "bid" else book.best_ask
        if best is None:
            return
        for d in self.distances:
            price = best - d if side == "bid" else best + d
            report = m.limit(self.cid, side, price, 1)
            if report is not None:
                self.posted[d] += 1
                if report["resting_quantity"]:
                    self._live.append((report["order_id"], d, self.account.shares))
                else:
                    self.filled[d] += 1


def fit_resilience(path: np.ndarray, duration: int, after: int) -> float:
    """Exponential rate of the decay of the paired impact after completion toward its final level."""
    tail = path[duration: duration + after + 1]
    if len(tail) < 10:
        return float("nan")
    final = tail[-max(5, len(tail) // 10):].mean()
    y = tail - final
    if y[0] <= 0:
        return float("nan")
    keep = y > 0.05 * y[0]
    t = np.arange(len(tail))[keep]
    if len(t) < 3:
        return float("nan")
    slope, _ = np.polyfit(t, np.log(y[keep]), 1)
    return float(max(-slope, 0.0))


def calibrate(seeds, fractions=(0.05, 0.1, 0.2), horizon: int = 2_000, burn_in: int = 2_000, after: int = 500,
              decision_interval: int = 5, base: SimConfig | None = None, verbose: bool = True) -> Calibration:
    cfg = adversary("A0", base=base)
    sigmas, depths, slips, rates, perm, rhos = [], [], [], [], [], []
    fill_posted = {d: 0 for d in (0, 1, 2, 3)}
    fill_filled = dict(fill_posted)
    window = horizon + after + 1
    for seed in seeds:
        # background path, depth, sigma, fill probabilities
        sim = Simulator(int(seed), cfg)
        prober = sim.add_external(FillProber, 10_000 * cfg.book.mid, 10_000, interval=decision_interval)
        sim.run(burn_in)
        ps, mids, depth = sim.market.public, np.empty(window), []
        for t in range(window):
            sim.step()
            mids[t] = ps.mid
            if t % 50 == 0 and ps.best_bid is not None and ps.best_ask is not None:
                depth.append(np.mean([sum(v for p, v in sim.book.levels(side)[:5]) for side in ("bid", "ask")]))
        sigmas.append(np.diff(mids).std())
        depths.append(np.mean(depth))
        for d in fill_posted:
            fill_posted[d] += prober.posted[d]
            fill_filled[d] += prober.filled[d]
        volume = expected_volume_from_tape(sim.market.tape, window)
        # TWAP-market runs of several sizes, paired against the background path
        for f in fractions:
            run = Simulator(int(seed), cfg)
            task = ExecutionTask(side="bid", size_fraction=f, horizon=horizon, decision_interval=decision_interval)
            executor = run.add_external(ExecutorAgent, 20_000 * cfg.book.mid, 20_000, task=task,
                                        strategy=TwapMarket(), start=burn_in, expected_volume=volume)
            run.run(burn_in)
            rps, rmids = run.market.public, np.empty(window)
            for t in range(window):
                run.step()
                rmids[t] = rps.mid
            r = executor.result
            if r is None or not r.executed:
                continue
            before = {}
            for step, price, qty, aggressive in r.fills:
                if aggressive and step - burn_in - 1 >= 0:
                    before.setdefault(step, rmids[step - burn_in - 1])
            slip = np.mean([price - before[step] for step, price, qty, agg in r.fills if agg and step in before])
            slips.append(slip)
            rates.append(r.quantity / horizon)
            impact = rmids - mids
            perm.append(impact[min(horizon, window - 1)] / r.quantity)
            rho = fit_resilience(impact, horizon, after)
            if not math.isnan(rho):
                rhos.append(rho)
            if verbose:
                print(f"[calibrate] seed {seed} Q/V={f:.2f} Q={r.quantity} slippage/child={slip:.2f} ticks "
                      f"impact@end={impact[horizon]:.1f} ticks rho={rho:.4f}", flush=True)
    eta = float(np.polyfit(rates, slips, 1)[0]) if len(set(rates)) > 1 else float(np.mean(slips) / np.mean(rates))
    fill_prob = tuple(fill_filled[d] / fill_posted[d] if fill_posted[d] else 0.0 for d in sorted(fill_posted))
    cal = Calibration(sigma=float(np.mean(sigmas)), temporary_impact=max(eta, 1e-6),
                      permanent_impact=float(np.mean(perm)), resilience=float(np.mean(rhos)) if rhos else 0.004,
                      depth=float(np.mean(depths)), fill_prob=fill_prob)
    if verbose:
        print(f"[calibrate] {cal}")
    return cal


def save(cal: Calibration, path) -> None:
    with open(path, "w") as f:
        json.dump(asdict(cal), f, indent=1)


def load(path) -> Calibration:
    with open(path) as f:
        d = json.load(f)
    return replace(Calibration(), **{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
