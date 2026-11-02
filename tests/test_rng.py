"""Counter-based draws: pure functions of their address, immune to changes in control flow."""

import pickle
from dataclasses import replace

import numpy as np
import pytest

from order_book.agents.market_maker import MarketMaker
from order_book.book import OrderBook
from order_book.config import PopulationConfig, SimConfig
from order_book.market import Market
from order_book.presets import adversary, private_valuations
from order_book.rng import NOISE_SIZE, VALUE, KeyedStream, keyed
from order_book.simulator import Simulator

COUNTER = replace(SimConfig(), rng="counter")


def test_a_draw_depends_only_on_its_address():
    s = KeyedStream(7, "MM1")
    first = s.at(100, 3).random(5).tolist()
    s.at(5, 1).normal(size=17)                      # any other use of the stream in between
    s.at(100, 4).integers(0, 10, 3)
    assert s.at(100, 3).random(5).tolist() == first
    assert KeyedStream(7, "MM1").at(100, 3).random(5).tolist() == first   # a fresh object
    assert s.at(100, 3, index=1).random(5).tolist() != first
    assert s.at(101, 3).random(5).tolist() != first
    assert s.at(100, 2).random(5).tolist() != first
    assert KeyedStream(8, "MM1").at(100, 3).random(5).tolist() != first
    assert KeyedStream(7, "MM2").at(100, 3).random(5).tolist() != first


def test_partial_word_buffers_do_not_leak_between_addresses():
    """A 32-bit integer draw leaves half a word buffered; the next address must not start from it."""
    s = KeyedStream(1, "x")
    clean = s.at(9, 1).integers(0, 2 ** 31, 4).tolist()
    s.at(9, 2).integers(0, 2 ** 31, 3)              # odd count: half a word left over
    assert s.at(9, 1).integers(0, 2 ** 31, 4).tolist() == clean


def test_arrays_are_stable_in_their_prefix():
    """A Poisson count that differs between two runs changes only the tail of an agent's order attributes."""
    s = KeyedStream(3, "Noise")
    assert s.at(4, NOISE_SIZE).random(3).tolist() == s.at(4, NOISE_SIZE).random(8).tolist()[:3]
    assert s.at(4, NOISE_SIZE).integers(1, 13, 3).tolist() == s.at(4, NOISE_SIZE).integers(1, 13, 8).tolist()[:3]
    assert s.at(4, NOISE_SIZE).geometric(0.25, 3).tolist() == s.at(4, NOISE_SIZE).geometric(0.25, 8).tolist()[:3]


def test_poisson_arrival_counts_are_monotone_in_the_rate():
    """Attention differs between paired runs; the same address must give ordered counts, not unrelated ones."""
    s = KeyedStream(5, "Noise")
    for step in range(500):
        low = s.at(step, 6).poisson(2.5)
        high = s.at(step, 6).poisson(3.5)
        assert low <= high


def _maker(rng):
    book = OrderBook(20, 10_000, 1)
    market = Market(book, analysis=False)
    return MarketMaker(0, market, rng, 10_000_000, 1000), market


@pytest.mark.parametrize("rng_factory, shifted", [
    (lambda: np.random.default_rng(0), True),
    (lambda: KeyedStream(0, "MM1"), False),
])
def test_one_extra_requote_shifts_every_later_draw_of_a_stateful_stream_but_not_a_keyed_one(rng_factory, shifted):
    """The defect of Buffalo, Pearson & Klein, on the agent they describe: a maker that requotes one extra time."""
    plain, market_a = _maker(rng_factory())
    extra, market_b = _maker(rng_factory())
    extra._draw_size("bid")                         # the intervention: one requote that the other run did not have
    a, b = [], []
    for step in range(1, 40):
        market_a.step = market_b.step = step
        a.append(plain._draw_size("bid"))
        b.append(extra._draw_size("bid"))
    assert (a != b) == shifted


def test_keyed_helper_passes_stateful_generators_through():
    rng = np.random.default_rng(0)
    assert keyed(rng, 5, VALUE) is rng
    stream = KeyedStream(0, "x")
    assert isinstance(keyed(stream, 5, VALUE), np.random.Generator)
    assert not hasattr(stream, "random")            # no unaddressed draws


def test_counter_mode_is_deterministic_snapshot_safe_and_conserves():
    a, b = Simulator(seed=3, config=COUNTER), Simulator(seed=3, config=COUNTER)
    a.run(400)
    b.run(400)
    assert a.market.tape == b.market.tape and len(a.market.tape) > 50
    a.check_conservation()
    clone = pickle.loads(pickle.dumps(a))
    a.run(200)
    clone.run(200)
    assert a.market.tape == clone.market.tape
    c, d = Simulator.restore(a.snapshot()), Simulator.restore(a.snapshot())
    c.reseed(9)
    d.reseed(9)
    c.run(150)
    d.run(150)
    assert c.market.tape == d.market.tape
    a.run(150)
    assert c.market.tape != a.market.tape


def test_counter_mode_runs_every_preset():
    for cfg in (adversary("A1"), adversary("A4"), private_valuations()):
        sim = Simulator(seed=2, config=replace(cfg, rng="counter"))
        sim.run(300)
        sim.check_conservation()


def test_unknown_rng_mode_is_rejected():
    with pytest.raises(ValueError, match="SimConfig.rng"):
        Simulator(seed=0, config=replace(SimConfig(), rng="philox"))


def test_counter_streams_are_keyed_by_who_not_by_position():
    """Removing agents does not change anyone else's draws; in stateful mode it shifts the spawn order."""
    full = Simulator(seed=4, config=COUNTER, analysis=False)
    fewer = Simulator(seed=4, config=replace(COUNTER, population=PopulationConfig(institutions=1, speculators=5)),
                      analysis=False)
    assert [m.base_size for m in full.agents[:4]] == [m.base_size for m in fewer.agents[:4]]
    noise = next(a for a in full.agents if type(a).__name__ == "NoiseFlow")
    noise2 = next(a for a in fewer.agents if type(a).__name__ == "NoiseFlow")
    assert noise.rng.at(10, 6).random() == noise2.rng.at(10, 6).random()
    assert full.fundamental.rng.at(10, VALUE).random() == fewer.fundamental.rng.at(10, VALUE).random()
    # An executor keeps its stream whether or not the population has an anticipator.
    a1 = Simulator(seed=4, config=replace(adversary("A1"), rng="counter"), analysis=False)
    a4 = Simulator(seed=4, config=replace(adversary("A4"), rng="counter"), analysis=False)
    assert a1.new_stream("Ext1").at(7, 21).random() == a4.new_stream("Ext1").at(7, 21).random()
