"""Execution task, executor agent, baseline strategies and calibration (plan sections 5.1 and 7)."""

from dataclasses import replace

import numpy as np
import pytest

from order_book.execution import (
    AlmgrenChriss, Calibration, Decision, ExecState, ExecutionTask, ExecutorAgent, ObizhaevaWang, Strategy,
    all_baselines, expected_volume_from_tape, make_strategy,
)
from order_book.execution.calibration import FillProber, fit_resilience, load, save
from order_book.presets import adversary
from order_book.simulator import Simulator

ENDOWMENT = 20_000


def episode(strategy, side="bid", size=0.1, horizon=200, interval=5, burn_in=500, seed=1, adv="A1", task_kwargs=None):
    sim = Simulator(seed, adversary(adv))
    sim.run(burn_in)
    volume = expected_volume_from_tape(sim.market.tape, burn_in)
    task = ExecutionTask(side=side, size_fraction=size, horizon=horizon, decision_interval=interval, **(task_kwargs or {}))
    executor = sim.add_external(ExecutorAgent, ENDOWMENT * 10_000, ENDOWMENT, task=task, strategy=strategy,
                                start=burn_in, expected_volume=volume)
    sim.run(horizon + 100)
    return sim, executor


@pytest.mark.parametrize("name", sorted(all_baselines()))
@pytest.mark.parametrize("side", ["bid", "ask"])
def test_every_baseline_completes_within_limits(name, side):
    sim, ex = episode(make_strategy(name), side)
    r = ex.result
    assert ex.done and r.end is not None and r.end <= ex.start + ex.task.horizon + ex.task.grace
    assert r.executed == r.quantity == ex.quantity and r.incomplete == 0
    assert r.shortfall_bps is not None and abs(r.shortfall_bps) < 100
    assert 0.0 <= r.completion <= 1.0 and 0 <= r.forced <= r.quantity
    assert r.aggressive_executed <= r.executed and r.orders_sent >= 1
    assert not sim.book.client_order_ids(ex.cid)
    delta = ex.account.shares - ENDOWMENT
    assert delta == (r.quantity if side == "bid" else -r.quantity)
    assert sum(q for _, _, q, _ in r.fills) == r.executed
    sim.check_conservation()


def test_shortfall_matches_the_tape():
    sim, ex = episode(make_strategy("twap_market"), "bid")
    fills = [(p, q) for step, p, q, agg, buyer, seller in sim.market.tape if buyer == ex.cid and step >= ex.start]
    paid, lots = sum(p * q for p, q in fills), sum(q for _, q in fills)
    r = ex.result
    assert lots == r.executed and paid == r.notional
    assert r.shortfall_bps == pytest.approx((paid / lots - r.arrival_mid) / r.arrival_mid * 1e4)


def test_immediate_sends_one_order_and_twap_one_per_decision():
    _, immediate = episode(make_strategy("immediate"))
    assert immediate.result.orders_sent == 1 and immediate.result.end <= immediate.start + 1
    _, twap = episode(make_strategy("twap_market"), horizon=200, interval=5)
    assert twap.result.orders_sent == min(twap.quantity, 40)  # one child per decision (40 decisions)


class DoNothing(Strategy):
    def decide(self, s, rng):
        return Decision()


def test_forced_completion_at_the_horizon_counts_in_full():
    _, ex = episode(DoNothing())
    r = ex.result
    assert r.forced == r.quantity == r.executed and r.completion == 0.0
    assert r.end == ex.start + ex.task.horizon
    assert r.shortfall_bps is not None


class ThreeLimits(Strategy):
    """Tries to rest three orders, one of them far through the opposite best."""

    def decide(self, s: ExecState, rng):
        if s.elapsed == 0:
            return Decision(limit_qty=1, limit_price=s.behind(3))
        if s.elapsed == s.decision_interval:
            return Decision(limit_qty=1, limit_price=s.behind(4))
        if s.elapsed == 2 * s.decision_interval:
            return Decision(limit_qty=1, limit_price=s.behind(5))
        return Decision()


def test_hard_limits_on_resting_orders_and_through_ticks():
    sim = Simulator(1, adversary("A0"))
    sim.run(500)
    task = ExecutionTask(side="bid", size_fraction=0.1, horizon=100, decision_interval=5, max_resting_orders=2,
                         max_through_ticks=2, max_participation=0.5)
    ex = sim.add_external(ExecutorAgent, ENDOWMENT * 10_000, ENDOWMENT, task=task, strategy=ThreeLimits(),
                          start=500, expected_volume=5.0)
    sim.run(16)
    assert len(sim.book.client_order_ids(ex.cid)) == 2  # the third limit was refused
    # a marketable limit is capped at opposite best + max_through_ticks
    state = ex.state()
    ex._apply(Decision(cancel=True, limit_qty=1, limit_price=state.opposite_best() + 50))
    orders = sim.book.get_client_orders(ex.cid)
    assert all(o["price"] <= state.opposite_best() + task.max_through_ticks for o in orders)
    # market orders are capped at max_participation x expected interval volume
    before = ex._executed()
    ex._apply(Decision(cancel=True, market_qty=1_000))
    assert ex._executed() - before <= max(1, round(0.5 * 5.0 * 5))


def test_hidden_executor_is_invisible_to_the_flow_window_but_not_to_volume():
    plain = Simulator(2, adversary("A0"))
    hidden = Simulator(2, adversary("A0"))
    plain.run(300)
    hidden.run(300)
    for sim in (plain, hidden):
        cid = sim.join(10_000 * 10_000, 10_000)
        sim._cid = cid
    hidden.market.hide(hidden._cid)
    for sim in (plain, hidden):
        sim.market.market(sim._cid, "bid", 20)
        sim.market.end_step()
    assert plain.market.public.step_volume == hidden.market.public.step_volume
    assert plain.market.public.buy_flow - hidden.market.public.buy_flow == 20
    assert len(plain.market.public.step_trades) > len(hidden.market.public.step_trades)
    assert len(plain.market.tape) == len(hidden.market.tape)  # the tape still records everything


def test_expected_volume_from_tape():
    tape = [(1, 100, 3, "buy", 1, 2), (2, 100, 5, "sell", 1, 2), (9, 100, 2, "buy", 1, 2), (10, 100, 4, "sell", 1, 2)]
    assert expected_volume_from_tape(tape, 10) == pytest.approx(1.4)
    assert expected_volume_from_tape(tape, 2) == pytest.approx(3.0)  # steps 9 and 10
    assert expected_volume_from_tape([], 10) == 5.0


# ----- model-based schedules and calibration ---------------------------------------------- #

def _state(quantity=1000, elapsed=0, executed=0, horizon=2000, interval=5):
    return ExecState(step=0, elapsed=elapsed, horizon=horizon, decision_interval=interval,
                     decisions_left=max(1, -(-(horizon - elapsed) // interval)), side="bid", quantity=quantity,
                     remaining=quantity - executed, executed=executed, aggressive_executed=0, arrival_mid=10_000,
                     mid=10_000, best_bid=9_999, best_ask=10_000, spread=1, imbalance=0.0,
                     public_volume_since_start=0, own_resting=0, own_resting_orders=[], expected_volume_per_step=5.0,
                     own_flow_share=0.0, public=None)


def test_almgren_chriss_schedule_is_front_loaded_and_ends_at_q():
    s = _state()
    fast, slow = AlmgrenChriss(1e-5), AlmgrenChriss(1e-7)
    assert fast.scheduled_executed_at(s, 2000) == pytest.approx(1000) == slow.scheduled_executed_at(s, 2000)
    assert fast.scheduled_executed_at(s, 500) > slow.scheduled_executed_at(s, 500) > 250 * 0.9
    sched = [fast.scheduled_executed_at(s, t) for t in range(0, 2001, 100)]
    assert all(b >= a for a, b in zip(sched, sched[1:]))


def test_obizhaeva_wang_blocks_and_rate_sum_to_q():
    ow = ObizhaevaWang(Calibration(resilience=0.002))
    s = _state()
    ow.reset(s, None)
    assert ow.block == pytest.approx(1000 / (0.002 * 2000 + 2))
    assert ow.block * 2 + ow.rate * 2000 == pytest.approx(1000)
    first = ow.decide(_state(elapsed=0), None)
    assert first.market_qty == -(-ow.block // 1)


def test_fit_resilience_recovers_the_decay_rate():
    t = np.arange(0, 2500)
    path = np.where(t < 2000, 10 * t / 2000, 4 + 6 * np.exp(-0.01 * (t - 2000)))
    assert fit_resilience(path, 2000, 499) == pytest.approx(0.01, rel=0.15)


def test_calibration_roundtrip_and_fill_prober(tmp_path):
    cal = replace(Calibration(), sigma=0.5, fill_prob=(0.3, 0.1, 0.0, 0.0))
    save(cal, tmp_path / "cal.json")
    assert load(tmp_path / "cal.json") == cal
    sim = Simulator(0, adversary("A0"))
    prober = sim.add_external(FillProber, 10_000 * 10_000, 10_000, interval=5)
    sim.run(600)
    assert prober.posted[0] > 50 and prober.filled[0] <= prober.posted[0]
    assert prober.filled[0] / prober.posted[0] > prober.filled[3] / max(prober.posted[3], 1)
    sim.check_conservation()
