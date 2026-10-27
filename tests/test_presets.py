"""Named configurations: adversary wiring, market regimes (E7), fundamental switch."""

from dataclasses import replace

import pytest

from order_book.config import SimConfig, market_makers
from order_book.presets import FLOW_THRESHOLDS, adversary, no_institutions, private_valuations, regime
from order_book.simulator import Simulator


def test_adversary_wires_detection_without_clobbering_the_market_maker_config():
    # a thin-book base: small maker quotes must survive the adversary's re-wiring
    base = regime(liquidity=0.5, institutions=1, makers=3, mm_size_scale=0.5)
    thin_size = base.population.market_makers[0].size
    assert thin_size[1] < SimConfig().population.market_makers[0].size[1]
    a0 = adversary("A0", base=base)
    a1 = adversary("A1", "strict", base=base)
    a4 = adversary("A4", "lenient", base=base, detecting=1)
    for cfg in (a0, a1, a4):
        assert len(cfg.population.market_makers) == 3
        assert all(m.size == thin_size for m in cfg.population.market_makers)  # size preserved
        assert cfg.population.institutions == 1 and cfg.population.speculators == base.population.speculators
    assert [m.detector for m in a0.population.market_makers] == [None, None, None]
    assert a0.population.anticipators == 0
    assert [m.detector for m in a1.population.market_makers] == ["flow", "flow", None]  # min(2, 3)
    assert a1.flow_detector.threshold == FLOW_THRESHOLDS["strict"]
    assert [m.detector for m in a4.population.market_makers] == ["flow", None, None]    # detecting=1
    assert a4.population.anticipators == 1 and a4.flow_detector.threshold == FLOW_THRESHOLDS["lenient"]


def test_regime_scales_liquidity_cover_and_book():
    default = SimConfig()
    thin = regime(liquidity=0.5, institutions=1, makers=2, mm_size_scale=0.5)
    assert thin.noise.rate == pytest.approx(default.noise.rate * 0.5)
    assert thin.population.speculators == round(default.population.speculators * 0.5)
    assert thin.population.institutions == 1 and len(thin.population.market_makers) == 2
    lo, hi = default.population.market_makers[0].size
    assert thin.population.market_makers[0].size == (max(1, round(lo * 0.25)), max(2, round(hi * 0.25)))
    assert regime().population.institutions == default.population.institutions  # identity but for detectors
    assert all(m.detector is None for m in regime().population.market_makers)   # detection wired by adversary
    assert regime().population.market_makers[0].size == default.population.market_makers[0].size
    assert no_institutions().population.institutions == 0


def test_a_thin_low_cover_regime_runs_and_conserves():
    cfg = adversary("A4", base=regime(liquidity=0.4, institutions=0, mm_size_scale=0.5))
    sim = Simulator(0, cfg, analysis=False)
    assert sim.config.population.institutions == 0 and not any(
        type(a).__name__ == "InstitutionalTrader" for a in sim.agents)
    sim.run(500)
    sim.check_conservation()
    # thinner market => lower traded volume than the deep-cover default on the same seed
    deep = Simulator(0, adversary("A4"), analysis=False)
    deep.run(500)
    assert sim.book.trade_id < deep.book.trade_id


def test_private_valuations_compose_with_a_regime():
    cfg = private_valuations(base=regime(liquidity=0.5, institutions=2))
    assert cfg.fundamental.private and cfg.population.institutions == 2
    assert cfg.noise.rate == pytest.approx(SimConfig().noise.rate * 0.5)
