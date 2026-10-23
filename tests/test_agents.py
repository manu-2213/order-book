"""Phase 1 agents: institutional metaorders, order anticipator, maker fragility, private valuations."""

from dataclasses import replace

import numpy as np
import pytest

from order_book.agents import InstitutionalTrader, MetaorderLog
from order_book.config import EXECUTION_STYLES, InstitutionalConfig, PopulationConfig, SimConfig, market_makers
from order_book.presets import adversary, private_valuations
from order_book.simulator import Simulator


def quiet_institutions(**kwargs) -> SimConfig:
    cfg = SimConfig()
    return replace(cfg, institution=replace(cfg.institution, start_rate=0.0, **kwargs))


@pytest.mark.parametrize("style", EXECUTION_STYLES)
@pytest.mark.parametrize("side", ["bid", "ask"])
def test_every_execution_style_completes_and_is_labelled(style, side):
    sim = Simulator(seed=1, config=quiet_institutions())
    trader = sim.add_external(InstitutionalTrader, 10_000 * 10_000, 10_000,
                              cfg=replace(InstitutionalConfig(), start_rate=0.0), log=sim.metaorders)
    sim.run(300)
    shares_before = trader.account.shares
    record = trader.begin(side, 400, 0.1, style, duration=400)
    assert record is sim.metaorders.records[-1]
    assert (record.side, record.style, record.total, record.start) == (side, style, 400, 300)
    sim.run(450)
    assert record.end is not None and record.end <= record.planned_end
    assert record.executed == 400
    delta = trader.account.shares - shares_before
    assert delta == (400 if side == "bid" else -400)
    assert record.notional > 0 and abs(record.avg_price - record.arrival_mid) < 50
    assert trader.record is None and not sim.book.client_order_ids(trader.cid)
    sim.check_conservation()


def test_metaorders_arrive_with_log_normal_sizes_and_balanced_sides():
    cfg = SimConfig()
    cfg = replace(cfg, institution=replace(cfg.institution, start_rate=0.01, duration=(50, 200)))
    sim = Simulator(seed=3, config=cfg)
    sim.run(4_000)
    records = [r for r in sim.metaorders.records if r.end is not None]
    assert len(records) >= 20
    sizes = np.array([r.total for r in records])
    assert sizes.min() < 300 < sizes.max() and sizes.max() <= cfg.institution.size_max
    assert {r.style for r in records} >= set(EXECUTION_STYLES) - {None}
    buys = sum(r.side == "bid" for r in records)
    assert 0.25 < buys / len(records) < 0.75
    assert all(0.02 <= r.participation <= 0.20 for r in records)
    assert all(r.executed <= r.total for r in records)
    assert Simulator(seed=3, config=cfg, analysis=False).metaorders.records is None  # training mode keeps no labels


def test_institutional_side_reverts_inventory():
    sim = Simulator(seed=0, config=quiet_institutions())
    trader = sim.agents[5]
    assert isinstance(trader, InstitutionalTrader)
    trader.account.shares = trader.endowment * 3  # heavily long: should overwhelmingly sell
    sides = []
    for _ in range(200):
        trader._start()
        sides.append(trader.record.side)
        trader.record, trader.resting = None, None
    assert sides.count("ask") > 170


def pin_signal(sim: Simulator, value: float) -> None:
    """Freeze the flow detector at ``value`` so the market keeps running while the adversary reacts."""
    detector = sim.detectors["flow"]
    detector.signal = value
    detector.update = lambda public: detector.signal


def test_order_anticipator_follows_signal_within_limits_and_unwinds():
    sim = Simulator(seed=0, config=adversary("A4"), analysis=False)
    sim.run(300)
    anticipator, cfg = sim.agents[-1], sim.agents[-1].cfg
    pin_signal(sim, 1.0)
    sim.run(150)
    position = anticipator.account.shares - anticipator.target
    assert 0 < position <= cfg.max_position
    pin_signal(sim, 0.0)
    sim.run(150)
    assert anticipator.account.shares == anticipator.target
    pin_signal(sim, -1.0)
    sim.run(150)
    assert -cfg.max_position <= anticipator.account.shares - anticipator.target < 0
    sim.check_conservation()


def test_contrarian_anticipator_trades_against_the_flow_first():
    cfg = adversary("A4")
    cfg = replace(cfg, anticipator=replace(cfg.anticipator, contrarian_first=True, switch_after=10))
    sim = Simulator(seed=0, config=cfg, analysis=False)
    sim.run(300)
    anticipator = sim.agents[-1]
    anticipator.confident_steps = 0
    pin_signal(sim, 1.0)
    sim.run(5)
    assert anticipator.account.shares < anticipator.target  # sold into the detected buying
    sim.run(80)
    assert anticipator.account.shares > anticipator.target  # then switched to buying with it


def test_market_maker_pulls_quotes_on_volatility_spike_and_cools_down():
    sim = Simulator(seed=0, analysis=False)
    sim.run(500)
    maker, ps = sim.agents[0], sim.market.public
    assert any(q is not None for q in maker.quotes.values())
    ps.var_fast, ps.var_slow = 100.0, 1.0  # ratio 10 > pull_ratio
    maker.act()
    assert maker.quotes == {"bid": None, "ask": None}
    assert not sim.book.client_order_ids(maker.cid)
    assert maker.pulled_until == sim.market.step + maker.cfg.cooldown
    ps.var_fast = 1.0
    maker.act()
    assert not sim.book.client_order_ids(maker.cid)  # still in cooldown
    sim.market.step = maker.pulled_until
    maker.act()
    assert any(q is not None for q in maker.quotes.values())


def test_market_maker_size_falls_with_relative_volatility():
    sim = Simulator(seed=0, analysis=False)
    sim.run(500)
    maker, ps = sim.agents[0], sim.market.public
    ps.var_fast, ps.var_slow = 1.0, 1.0
    maker.pulled_until = 0
    maker.base_size = {"bid": 20, "ask": 20}
    sim.market.cancel_all(maker.cid)
    maker.quotes = {"bid": None, "ask": None}
    maker.act()
    full = sum(o["quantity"] for o in sim.book.get_client_orders(maker.cid))
    ps.var_fast = 4.0  # ratio 2 -> scale 1 - fragility * 1
    sim.market.cancel_all(maker.cid)
    maker.quotes = {"bid": None, "ask": None}
    maker.base_size = {"bid": 20, "ask": 20}
    maker.act()
    reduced = sum(o["quantity"] for o in sim.book.get_client_orders(maker.cid))
    assert 0 < reduced < full


def test_private_valuations_mode_runs_and_differs_from_shared_mode():
    shared = Simulator(seed=2, analysis=False)
    private = Simulator(seed=2, config=private_valuations(), analysis=False)
    assert private.speculators.private and private.speculators.values.shape == (40,)
    shared.run(500)
    private.run(500)
    private.check_conservation()
    assert private.book.order_id != shared.book.order_id  # the two information models produce different flow
    assert np.all(np.abs(private.speculators.values - private.market.public.mid) < 200)


def test_external_agent_leaves_background_draws_unchanged():
    """The CRN property behind paired counterfactuals. With an idle external trader present, every
    background stream and the event log are identical to the run without it. Once it trades, the
    scheduler and hidden-value streams stay identical (they do not depend on market state); the agent
    streams stay shared up to state-dependence (e.g. the number of noise orders depends on attention)."""
    plain = Simulator(seed=7)
    treated = Simulator(seed=7)
    treated.add_external(InstitutionalTrader, 10_000 * 10_000, 10_000,
                         cfg=replace(InstitutionalConfig(), start_rate=0.0), log=treated.metaorders)
    plain.run(300)
    treated.run(300)
    assert plain.market.tape == treated.market.tape
    for a, b in zip(plain.streams, treated.streams):
        assert a.bit_generator.state["state"] == b.bit_generator.state["state"]
    treated.externals[0].begin("bid", 300, 0.1, "twap_market", duration=200)
    plain.run(300)
    treated.run(300)
    for i in (0, 1):  # scheduler, fundamental
        assert plain.streams[i].bit_generator.state["state"] == treated.streams[i].bit_generator.state["state"]
    assert plain.fundamental.value == treated.fundamental.value
    assert treated.market.public.mid != plain.market.public.mid  # the market reacted to the order
    treated.check_conservation()


def test_adversary_presets():
    a0, a1, a4 = adversary("A0"), adversary("A1", "strict"), adversary("A4", "lenient")
    assert all(m.detector is None for m in a0.population.market_makers) and a0.population.anticipators == 0
    assert [m.detector for m in a1.population.market_makers] == ["flow", "flow", None, None]
    assert a1.flow_detector.threshold == 0.45 and a4.flow_detector.threshold == 0.2
    assert a4.population.anticipators == 1
    with pytest.raises(ValueError):
        adversary("A2")
    assert market_makers(3, 3)[2].detector == "flow"
