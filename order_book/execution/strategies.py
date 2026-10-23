"""Baseline execution strategies (plan section 7): classical schedules, model-based schedules and stealth
heuristics. Every strategy works through the same ``ExecutorAgent``, limits and forced completion.

Model-based strategies read their fitted parameters from a ``Calibration``
(``order_book.execution.calibration``); the defaults are the values measured
on the default market so that the strategies run without a calibration file,
and ``scripts/calibrate.py`` refreshes them.
"""

import math
from dataclasses import dataclass, field

from order_book.config import EXECUTION_STYLES
from order_book.execution.task import Decision, ExecState, Strategy


@dataclass(frozen=True)
class Calibration:
    """Market parameters fitted under A0 (see ``calibration.py``); units are ticks, lots and steps."""

    sigma: float = 0.19                  # std of 1-step mid changes (ticks)
    temporary_impact: float = 0.035      # eta: child slippage against the mid per (lot/step) of trading rate
    permanent_impact: float = 0.0017     # mid change at completion per lot (ticks)
    resilience: float = 0.0028           # rho: exponential decay rate of transient impact per step
    depth: float = 227.0                 # lots resting within 5 ticks of the best (one side)
    fill_prob: tuple[float, ...] = (0.28, 0.015, 0.001, 0.0)  # P(limit at 0,1,2,3 ticks behind own best fills in N=5 steps)


# ----- classical schedules ---------------------------------------------------------------- #

class Immediate(Strategy):
    """Whole order by market order at the first decision (upper bound on impact, no timing risk)."""

    name = "immediate"

    def decide(self, s: ExecState, rng) -> Decision:
        return Decision(market_qty=s.remaining)


class TwapMarket(Strategy):
    """Equal slices at equal intervals, market orders."""

    name = "twap_market"

    def decide(self, s: ExecState, rng) -> Decision:
        return Decision(market_qty=math.ceil(s.remaining / s.decisions_left))


class TwapPassive(Strategy):
    """Each slice posted at own best; at the next decision the unfilled part of the slice is crossed."""

    name = "twap_passive"

    def reset(self, s: ExecState, rng) -> None:
        self.target = 0

    def decide(self, s: ExecState, rng) -> Decision:
        catch_up = max(0, self.target - s.executed)
        slice_qty = math.ceil(max(0, s.remaining - catch_up) / s.decisions_left)
        self.target = s.executed + catch_up + slice_qty
        return Decision(cancel=True, market_qty=catch_up, limit_qty=slice_qty)


class TwapRandom(Strategy):
    """TWAP with slice sizes jittered by +/-``jitter`` and decisions skipped at random (timing jitter)."""

    name = "twap_random"

    def __init__(self, jitter: float = 0.5, skip_prob: float = 0.3):
        self.jitter, self.skip_prob = jitter, skip_prob

    def decide(self, s: ExecState, rng) -> Decision:
        if s.decisions_left > 1 and rng.random() < self.skip_prob:
            return Decision()
        qty = math.ceil(s.remaining / s.decisions_left * rng.uniform(1 - self.jitter, 1 + self.jitter))
        return Decision(market_qty=qty)


class Pov(Strategy):
    """Trades a fixed share of observed market volume, catching up over the final tenth of the horizon."""

    name = "pov"

    def __init__(self, rate: float = 0.10):
        self.rate = rate

    def decide(self, s: ExecState, rng) -> Decision:
        if s.elapsed >= 0.9 * s.horizon:
            return Decision(market_qty=math.ceil(s.remaining / s.decisions_left))
        target = min(s.quantity, math.floor(self.rate * s.public_volume_since_start))
        return Decision(market_qty=max(0, target - s.executed))


# ----- model-based schedules -------------------------------------------------------------- #

class _ScheduleFollower(Strategy):
    """Follow a target trajectory ``scheduled_executed(t)`` with the passive-with-catch-up tactic."""

    def scheduled_executed(self, s: ExecState) -> float:
        raise NotImplementedError

    def decide(self, s: ExecState, rng) -> Decision:
        behind = max(0, math.floor(self.scheduled_executed(s)) - s.executed)
        next_target = self.scheduled_executed_at(s, s.elapsed + s.decision_interval)
        slice_qty = max(0, math.ceil(next_target) - s.executed - behind)
        return Decision(cancel=True, market_qty=behind, limit_qty=slice_qty)

    def scheduled_executed_at(self, s: ExecState, elapsed: int) -> float:
        raise NotImplementedError


class AlmgrenChriss(_ScheduleFollower):
    """Almgren-Chriss trajectory with risk aversion ``risk_aversion`` (lambda), executed passively with
    catch-up: remaining(t) = Q sinh(kappa (T - t)) / sinh(kappa T), kappa = sqrt(lambda sigma^2 / eta)."""

    name = "almgren_chriss"

    def __init__(self, risk_aversion: float = 1e-6, calibration: Calibration | None = None):
        self.risk_aversion = risk_aversion
        self.cal = calibration or Calibration()
        self.name = f"almgren_chriss_{risk_aversion:g}"

    def _kappa(self) -> float:
        return math.sqrt(self.risk_aversion * self.cal.sigma ** 2 / max(self.cal.temporary_impact, 1e-9))

    def scheduled_executed_at(self, s: ExecState, elapsed: int) -> float:
        k, T = self._kappa(), s.horizon
        t = min(elapsed, T)
        if k * T < 1e-6:
            return s.quantity * t / T
        return s.quantity * (1 - math.sinh(k * (T - t)) / math.sinh(k * T))

    def scheduled_executed(self, s: ExecState) -> float:
        return self.scheduled_executed_at(s, s.elapsed)


class ObizhaevaWang(Strategy):
    """Obizhaeva-Wang optimal schedule in a book that refills at rate ``rho``: a block of
    x0 = Q / (rho T + 2) at the start and at the end, and a constant rate rho x0 in between."""

    name = "obizhaeva_wang"

    def __init__(self, calibration: Calibration | None = None):
        self.cal = calibration or Calibration()

    def reset(self, s: ExecState, rng) -> None:
        rho, T = self.cal.resilience, s.horizon
        self.block = s.quantity / (rho * T + 2)
        self.rate = rho * self.block

    def decide(self, s: ExecState, rng) -> Decision:
        if s.elapsed == 0:
            target = self.block
        elif s.decisions_left <= 1:
            target = s.quantity
        else:
            target = self.block + self.rate * s.elapsed
        return Decision(market_qty=max(0, math.ceil(min(target, s.quantity)) - s.executed))


class LimitMarket(Strategy):
    """Limit-and-market policy in the spirit of Gueant-Lehalle-Fernandez-Tapia / Cartea-Jaimungal.

    Posts at the deepest distance d whose fitted fill probability per decision still gets the order
    done in the decisions left (``fill_prob[d] * decisions_left >= urgency``), keeps the resting order
    while its price is still the chosen one (queue position), lets execution lag the linear schedule
    by up to 1 / fill_prob[d] slices before catching up with market orders, and switches to market
    orders over the last two decisions.
    """

    name = "limit_market"

    def __init__(self, urgency: float = 1.5, calibration: Calibration | None = None):
        self.urgency = urgency
        self.cal = calibration or Calibration()

    def decide(self, s: ExecState, rng) -> Decision:
        slice_qty = math.ceil(s.remaining / s.decisions_left)
        if s.decisions_left <= 2:
            return Decision(cancel=True, market_qty=s.remaining if s.decisions_left <= 1 else slice_qty)
        distance, p = 0, self.cal.fill_prob[0]
        for d, pd in enumerate(self.cal.fill_prob):
            if pd * s.decisions_left >= self.urgency:
                distance, p = d, pd
        scheduled = s.quantity * s.elapsed / s.horizon
        slack = slice_qty / max(p, 1e-6)
        behind = max(0, math.floor(scheduled - slack) - s.executed)
        price = s.behind(distance)
        keep = bool(s.own_resting_orders) and all(o["price"] == price for o in s.own_resting_orders)
        want = min(s.remaining - behind, 2 * slice_qty) - (s.own_resting if keep else 0)
        return Decision(cancel=not keep, market_qty=behind, limit_qty=max(0, want), limit_price=price)


# ----- stealth heuristics ----------------------------------------------------------------- #

class Footprint(Strategy):
    """Footprint-aware rule: slow down when own share of the public signed flow is high, speed up when
    the natural flow goes the executor's way (cover), randomise sizes and timing, catch up near the end."""

    name = "footprint"

    def __init__(self, share_threshold: float = 0.3, flow_boost: float = 1.5, jitter: float = 0.4,
                 passive_share: float = 0.5):
        self.share_threshold, self.flow_boost = share_threshold, flow_boost
        self.jitter, self.passive_share = jitter, passive_share

    def decide(self, s: ExecState, rng) -> Decision:
        if s.elapsed >= 0.85 * s.horizon:
            return Decision(cancel=True, market_qty=math.ceil(s.remaining / s.decisions_left))
        base = s.remaining / s.decisions_left
        scale = rng.uniform(1 - self.jitter, 1 + self.jitter)
        if s.own_flow_share > self.share_threshold:
            scale *= 0.25                                   # we are too visible: back off
        if s.sign * s.imbalance > 0.2:
            scale *= self.flow_boost                        # others trade our way: hide in it
        qty = max(0, math.ceil(base * scale))
        passive = math.ceil(qty * self.passive_share)
        return Decision(cancel=True, market_qty=qty - passive, limit_qty=passive)


class Mimic(Strategy):
    """Copies the institutional population: draws one of its execution styles per episode and runs it
    with the institutional child interval, so the executor looks like one more institution."""

    name = "mimic"

    def __init__(self, styles=EXECUTION_STYLES, jitter: float = 0.5, front_load: float = 2.0, pov_rate=(0.05, 0.2)):
        self.styles, self.jitter, self.front_load, self.pov_rate = styles, jitter, front_load, pov_rate

    def reset(self, s: ExecState, rng) -> None:
        self.style = self.styles[int(rng.integers(len(self.styles)))]
        self.rate = float(rng.uniform(*self.pov_rate))
        self.target = 0
        self.skip_until = 0

    def decide(self, s: ExecState, rng) -> Decision:
        if self.style == "twap_market":
            return TwapMarket.decide(self, s, rng)
        if self.style == "twap_random":
            if s.elapsed < self.skip_until and s.decisions_left > 1:
                return Decision()
            qty = math.ceil(s.remaining / s.decisions_left * rng.uniform(1 - self.jitter, 1 + self.jitter))
            self.skip_until = s.elapsed + round(s.decision_interval * rng.uniform(1 - self.jitter, 1 + self.jitter))
            return Decision(market_qty=qty)
        if self.style == "twap_passive":
            return TwapPassive.decide(self, s, rng)
        if self.style == "pov":
            if s.elapsed >= 0.9 * s.horizon:
                return Decision(market_qty=math.ceil(s.remaining / s.decisions_left))
            target = min(s.quantity, math.floor(self.rate * s.public_volume_since_start))
            return Decision(market_qty=max(0, target - s.executed))
        k, T = self.front_load / s.horizon, s.horizon
        t = min(T, s.elapsed + s.decision_interval)
        scheduled = s.quantity * (1 - math.sinh(k * (T - t)) / math.sinh(k * T))
        return Decision(market_qty=max(0, math.ceil(scheduled) - s.executed))


# ----- registry ------------------------------------------------------------------------- #

def classic_strategies(calibration: Calibration | None = None, pov_rate: float = 0.10) -> dict[str, Strategy]:
    """The classical schedules of the plan, by name."""
    return {
        "immediate": Immediate(), "twap_market": TwapMarket(), "twap_passive": TwapPassive(),
        "twap_random": TwapRandom(), "pov": Pov(pov_rate),
    }


def model_strategies(calibration: Calibration | None = None,
                     risk_aversions=(1e-7, 1e-6, 1e-5)) -> dict[str, Strategy]:
    """Almgren-Chriss at three risk aversions (kappa T of roughly 0.6, 2 and 6 with the default
    calibration), Obizhaeva-Wang and the limit-and-market policy."""
    cal = calibration or Calibration()
    out: dict[str, Strategy] = {AlmgrenChriss(l, cal).name: AlmgrenChriss(l, cal) for l in risk_aversions}
    out["obizhaeva_wang"] = ObizhaevaWang(cal)
    out["limit_market"] = LimitMarket(calibration=cal)
    return out


def stealth_strategies() -> dict[str, Strategy]:
    return {"footprint": Footprint(), "mimic": Mimic()}


def all_baselines(calibration: Calibration | None = None) -> dict[str, Strategy]:
    return {**classic_strategies(calibration), **model_strategies(calibration), **stealth_strategies()}


def make_strategy(name: str, calibration: Calibration | None = None) -> Strategy:
    strategies = all_baselines(calibration)
    if name not in strategies:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(strategies)}")
    return strategies[name]
