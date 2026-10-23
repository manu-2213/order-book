from dataclasses import FrozenInstanceError, replace

import pytest

from order_book.config import BookConfig, PopulationConfig, SimConfig, market_makers
from order_book.detectors import FlowDetector
from order_book.simulator import Simulator
from tests.helpers import assert_reservations_consistent


def test_simulator_is_deterministic_conserves_wealth_and_restores():
    a, b = Simulator(seed=3, analysis=False), Simulator(seed=3, analysis=False)
    a.run(300)
    b.run(300)
    assert a.market.public.mid == b.market.public.mid
    assert a.book.trade_id == b.book.trade_id
    a.check_conservation()
    assert_reservations_consistent(a.market)
    assert a.market.tape is None
    clone = Simulator.restore(a.snapshot())
    clone.run(200)
    a.run(200)
    assert clone.market.public.mid == a.market.public.mid
    assert clone.book.order_id == a.book.order_id
    clone.reseed(7)
    a.reseed(8)
    clone.run(200)
    a.run(200)
    assert clone.book.order_id != a.book.order_id
    clone.check_conservation()


def test_analysis_mode_keeps_tape_and_history():
    sim = Simulator(seed=0)
    sim.run(100)
    assert sim.market.tape and len(sim.fundamental.history) == 101
    assert len(sim.book.trades) == len(sim.market.tape)
    sim.check_conservation()


def test_population_follows_config_and_detectors_are_shared():
    cfg = replace(SimConfig(), population=PopulationConfig(market_makers=market_makers(3, detecting=1),
                                                           institutions=1, speculators=5, anticipators=1))
    sim = Simulator(seed=0, config=cfg, analysis=False)
    makers = sim.agents[:3]
    flow = sim.detectors["flow"]
    assert [m.detector is flow for m in makers] == [True, False, False]
    assert list(sim.groups) == ["MM1", "MM2", "MM3", "Noise", "Inst1", "Speculators", "Antic1"]
    assert len(sim.market.accounts) == 3 + 1 + 1 + 5 + 1
    assert sim.agents[-1].detector is flow
    with pytest.raises(KeyError):
        Simulator(seed=0, config=replace(cfg, anticipator=replace(cfg.anticipator, detector="nope")))
    sim.run(50)
    sim.check_conservation()


def test_config_is_frozen_and_replaceable():
    cfg = SimConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.book.num_levels = 5
    variant = replace(cfg, book=replace(cfg.book, num_levels=5))
    assert (variant.book.num_levels, cfg.book.num_levels) == (5, 20)
    assert Simulator(seed=0, config=variant, analysis=False).book.num_levels == 5
    assert BookConfig() == cfg.book


def test_flow_detector_thresholds_and_saturates():
    class Public:
        imbalance = 1.0

    detector = FlowDetector()
    assert detector.update(Public()) == 0.0  # one step of memory is below the threshold
    for _ in range(100):
        signal = detector.update(Public())
    assert signal == pytest.approx(1.0)
    assert (detector.p_buy, detector.p_sell) == pytest.approx((1.0, 0.0))
    Public.imbalance = -1.0
    for _ in range(100):
        signal = detector.update(Public())
    assert signal == pytest.approx(-1.0)
