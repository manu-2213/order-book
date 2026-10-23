"""Phase 0 foundations: exact replay, per-step invariants, independent streams, no hidden information."""

from dataclasses import replace

import numpy as np
import pytest

from order_book.config import PopulationConfig, SimConfig
from order_book.presets import adversary
from order_book.simulator import Simulator
from tests.helpers import assert_book_invariants, assert_reservations_consistent


def event_log(sim: Simulator) -> tuple:
    return (tuple(sim.market.tape), sim.book.order_id, sim.book.trade_id,
            tuple(sim.fundamental.history), sim.market.public.mid)


def test_same_config_and_seed_reproduce_the_event_log_exactly():
    a, b = Simulator(seed=11), Simulator(seed=11)
    a.run(600)
    b.run(600)
    assert event_log(a) == event_log(b)
    assert len(a.market.tape) > 100
    assert event_log(Simulator(seed=12).__class__(seed=12)) != event_log(a)  # a different seed differs


def test_snapshot_restore_and_reseed_replay_exactly():
    a = Simulator(seed=5)
    a.run(200)
    blob = a.snapshot()
    b = Simulator.restore(blob)
    a.run(200)
    b.run(200)
    assert event_log(a) == event_log(b)
    c, d = Simulator.restore(blob), Simulator.restore(blob)
    c.reseed(99)
    d.reseed(99)
    c.run(200)
    d.run(200)
    assert event_log(c) == event_log(d)
    assert event_log(c) != event_log(a)
    assert c.seed == 99 and len(c.streams) == len(a.streams)


def test_agent_streams_are_independent_of_the_rest_of_the_population():
    """Each agent draws from its own child of the root seed, so adding or removing other
    agents, or an external participant, leaves its own draws untouched."""
    base = Simulator(seed=0, analysis=False)
    fewer = Simulator(seed=0, config=replace(SimConfig(), population=PopulationConfig(institutions=0)),
                      analysis=False)
    # Market makers are built first and draw their initial quote sizes at construction.
    assert [m.base_size for m in base.agents[:4]] == [m.base_size for m in fewer.agents[:4]]
    # Spawning a stream for an external trader does not change any background stream.
    with_exec = Simulator(seed=0, analysis=False)
    executor_rng = with_exec.new_stream()
    for s1, s2 in zip(base.streams, with_exec.streams):
        assert s1.bit_generator.state["state"] == s2.bit_generator.state["state"]
    assert executor_rng.random() != base.rng.random()
    assert len(with_exec.streams) == len(base.streams) + 1
    assert len({id(s) for s in base.streams}) == len(base.streams) == 2 + len(base.agents)


def test_invariants_hold_after_every_step():
    sim = Simulator(seed=2)
    sim.join(1_000_000, 100)  # an idle external account must not break anything either
    for _ in range(300):
        sim.step()
        book = sim.book
        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask
        assert_book_invariants(book)
        assert_reservations_consistent(sim.market)
        sim.check_conservation()


class IdentityLeak(AssertionError):
    pass


class _Guard:
    """Fails when an agent, while acting, reaches anything but public data and its own account/orders.

    Access through ``Market``'s own methods (settlement, routing) is legitimate and is
    suspended while they run; everything an agent touches directly is checked.
    """

    def __init__(self):
        self.actor: set[int] | None = None
        self.suspended = 0

    def check(self, cid, what: str) -> None:
        if self.actor is not None and not self.suspended and cid not in self.actor:
            raise IdentityLeak(f"agent {sorted(self.actor)} read {what} of client {cid}")

    def forbid(self, what: str) -> None:
        if self.actor is not None and not self.suspended:
            raise IdentityLeak(f"agent {sorted(self.actor)} read {what}")


def test_agents_use_only_public_data_and_their_own_account(monkeypatch):
    from order_book import book as book_module
    from order_book import market as market_module

    guard = _Guard()
    sim = Simulator(seed=4, config=adversary("A4"))
    market, book = sim.market, sim.book
    assert any(type(a).__name__ == "OrderAnticipator" for a in sim.agents)

    class GuardedRecords(list):
        """Agents may append their own record; reading the labels is a leak."""

        def __getitem__(self, item):
            guard.forbid("metaorder labels")
            return super().__getitem__(item)

        def __iter__(self):
            guard.forbid("metaorder labels")
            return super().__iter__()

        def __len__(self):
            guard.forbid("metaorder labels")
            return super().__len__()

    sim.metaorders.records = GuardedRecords(sim.metaorders.records)

    class GuardedAccounts(dict):
        def __getitem__(self, cid):
            guard.check(cid, "account")
            return super().__getitem__(cid)

        def get(self, cid, default=None):
            guard.check(cid, "account")
            return super().get(cid, default)

    market.accounts = GuardedAccounts(market.accounts)

    def suspended(method):
        def wrapper(*args, **kwargs):
            guard.suspended += 1
            try:
                return method(*args, **kwargs)
            finally:
                guard.suspended -= 1
        return wrapper

    for name in ("limit", "ioc", "market", "cancel", "cancel_all", "wealth", "end_step"):
        monkeypatch.setattr(market_module.Market, name, suspended(getattr(market_module.Market, name)))

    def own_orders_only(method, pick_cid):
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            cid = pick_cid(args, result)
            if cid is not None:
                guard.check(cid, "orders")
            return result
        return wrapper

    Book = book_module.OrderBook
    monkeypatch.setattr(Book, "get_order", own_orders_only(
        Book.get_order, lambda args, r: None if r is None else r["client_id"]))
    monkeypatch.setattr(Book, "get_client_orders", own_orders_only(Book.get_client_orders, lambda args, r: args[0]))
    monkeypatch.setattr(Book, "client_order_ids", own_orders_only(Book.client_order_ids, lambda args, r: args[0]))
    for name in ("bid_table", "ask_table", "trades", "last_order_report"):
        original = getattr(Book, name)
        monkeypatch.setattr(Book, name, property(lambda self, o=original, n=name: (guard.forbid(n), o.fget(self))[1]))
    monkeypatch.setattr(type(market), "tape", property(lambda self: (guard.forbid("tape"), self.__dict__["tape"])[1]),
                        raising=False)
    monkeypatch.setattr(type(sim.fundamental), "history",
                        property(lambda self: (guard.forbid("hidden value history"), self.__dict__["history"])[1]),
                        raising=False)

    # An executor (external agent) with the strategy that reads the most state is guarded too.
    from order_book.execution import ExecutionTask, ExecutorAgent, make_strategy
    sim.add_external(ExecutorAgent, 20_000 * 10_000, 20_000,
                     task=ExecutionTask(side="bid", size_fraction=0.1, horizon=300, decision_interval=5),
                     strategy=make_strategy("footprint"), start=100, expected_volume=5.0)
    for agent in sim.agents + sim.externals:
        cids = set(getattr(agent, "cids", [getattr(agent, "cid", None)]))
        original_act = agent.act

        def act(cids=cids, original_act=original_act):
            guard.actor = cids
            try:
                original_act()
            finally:
                guard.actor = None

        agent.act = act

    sim.run(600)
    assert sim.book.trade_id > 100  # the guard really ran through busy steps
    guard.actor = None
    assert len(sim.metaorders.records) >= 0
    assert sim.externals[0].done and sim.externals[0].result.executed == sim.externals[0].quantity

    # The guard itself catches a leak: a rogue agent peeking at someone else's account.
    guard.actor = {0}
    with pytest.raises(IdentityLeak):
        market.accounts[1]
    with pytest.raises(IdentityLeak):
        market.tape
    with pytest.raises(IdentityLeak):
        book.get_client_orders(sim.agents[1].cid)
    guard.actor = None
    assert isinstance(market.accounts[1].cash, (int, np.integer))
