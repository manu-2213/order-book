"""Paired impact measurement: one injected metaorder against the identical market without it.

Because every background agent has its own random stream and external agents
act outside the scheduler's permutation, a run with an injected metaorder and
the same seed without it share every background draw. Their mid-price
difference is the metaorder's impact (and the market's reaction to it) with
the market's own randomness removed, which is far more precise than reading
impact off the raw price path.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from order_book.agents import InstitutionalTrader
from order_book.config import InstitutionalConfig, SimConfig
from order_book.simulator import Simulator
from order_book.validation import fit_power_law


@dataclass
class ImpactRun:
    seed: int
    side: str
    style: str
    size: int
    executed: int
    participation: float
    duration: int
    q_over_v: float
    path: np.ndarray        # signed impact in ticks, from the start step, length ``horizon``
    shortfall_bps: float | None

    @property
    def end_ticks(self) -> float:
        return float(self.path[min(self.duration, len(self.path) - 1)])

    @property
    def peak_ticks(self) -> float:
        return float(self.path[: self.duration + 1].max())

    def residual_ticks(self, after: int) -> float:
        return float(self.path[min(self.duration + after, len(self.path) - 1)])


def _mid_path(sim: Simulator, steps: int) -> np.ndarray:
    ps, out = sim.market.public, np.empty(steps)
    for t in range(steps):
        sim.step()
        out[t] = ps.mid
    return out


def baseline(seed: int, config: SimConfig | None, burn_in: int, horizon: int) -> tuple[np.ndarray, float]:
    """Mid path after burn-in and the mean market volume per step over it, without the injected order."""
    sim = Simulator(seed, config, analysis=True)
    sim.run(burn_in)
    offset = len(sim.market.tape)
    mid = _mid_path(sim, horizon)
    volume = sum(t[2] for t in sim.market.tape[offset:]) / horizon
    return mid, volume


def injected(seed: int, config: SimConfig | None, burn_in: int, horizon: int, side: str, size: int,
             participation: float, style: str, duration: int, cash: int, shares: int) -> tuple[np.ndarray, object]:
    sim = Simulator(seed, config, analysis=True)
    trader = sim.add_external(InstitutionalTrader, cash, shares,
                              cfg=replace(InstitutionalConfig(), start_rate=0.0), log=sim.metaorders)
    sim.run(burn_in)
    record = trader.begin(side, size, participation, style, duration=duration)
    mid = _mid_path(sim, horizon)
    if record is not None and record.end is None:  # still running at the end of the window
        trader._finish()
    return mid, record


def measure_impact(seeds, fractions=(0.01, 0.02, 0.05, 0.1, 0.2), styles=("twap_market",),
                   config: SimConfig | None = None, burn_in: int = 2_000,
                   reference_horizon: int = 2_000, after: int = 500, side: str = "bid",
                   cash: int | None = None, shares: int = 20_000) -> list[ImpactRun]:
    """Inject metaorders of ``fractions`` x (volume per step x reference_horizon) lots, each executed over
    ``reference_horizon`` steps (so its participation rate is the fraction), and difference their mid
    path against the same seed without them."""
    runs = []
    horizon = reference_horizon + after + 1
    mid0 = (config or SimConfig()).book.mid
    for seed in seeds:
        base, volume = baseline(int(seed), config, burn_in, horizon)
        for style in styles:
            for fraction in fractions:
                size = max(1, round(fraction * volume * reference_horizon))
                mid, record = injected(int(seed), config, burn_in, horizon, side, size, fraction, style,
                                       reference_horizon, cash if cash is not None else shares * mid0, shares)
                if record is None:
                    continue
                sign = 1 if side == "bid" else -1
                runs.append(ImpactRun(int(seed), side, style, size, record.executed, fraction,
                                      record.end - record.start, record.executed / (volume * reference_horizon),
                                      sign * (mid - base), record.shortfall_bps))
    return runs


def summarise_impact(runs: list[ImpactRun], after: int = 500) -> dict:
    q = [r.q_over_v for r in runs]
    end = [r.end_ticks for r in runs]
    peak = [r.peak_ticks for r in runs]
    peak_sum = sum(peak)
    by_size: dict = {}
    for r in runs:
        by_size.setdefault(round(r.participation, 3), []).append(r)
    fit_end, fit_peak = fit_power_law(q, end, bins=len(by_size)), fit_power_law(q, peak, bins=len(by_size))
    return {
        "n": len(runs), "delta_end": fit_end["delta"], "delta_peak": fit_peak["delta"],
        "bins_end": fit_end["bins"], "bins_peak": fit_peak["bins"],
        "end_share_of_peak": sum(end) / peak_sum if peak_sum > 0 else math.nan,
        "remaining_share_of_peak": sum(r.residual_ticks(after) for r in runs) / peak_sum if peak_sum > 0 else math.nan,
        "mean_shortfall_bps": float(np.mean([r.shortfall_bps for r in runs if r.shortfall_bps is not None]))
        if runs else math.nan,
        "per_fraction": {k: {"n": len(v), "mean_end": float(np.mean([r.end_ticks for r in v])),
                             "sd_end": float(np.std([r.end_ticks for r in v])),
                             "mean_peak": float(np.mean([r.peak_ticks for r in v]))} for k, v in sorted(by_size.items())},
    }
