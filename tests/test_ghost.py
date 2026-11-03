"""The ghost market's matching rule (R1-R5 in ``order_book.ghost``), one rule per test."""

import numpy as np
import pytest

from order_book.book import OrderBook
from order_book.execution import ExecutionTask, make_strategy
from order_book.ghost import GhostConfig, GhostMarket
from order_book.market import Market
from tests.helpers import assert_accounted, assert_book_invariants, assert_reservations_consistent

EXECUTOR = 9


def source_market(rests=()) -> Market:
    market = Market(OrderBook(num_levels=20, mid_price=100, distance_from_mid=1), analysis=False)
    for cid in (1, 2, 3):
        market.open_account(cid, 10 ** 7, 10 ** 5)
    for cid, side, price, quantity in rests:
        market.limit(cid, side, price, quantity)
    return market


def fork(source: Market, ttl: int | None = 50) -> GhostMarket:
    ghost = GhostMarket(source, GhostConfig(orphan_ttl=ttl))
    ghost.open_account(EXECUTOR, 10 ** 7, 10 ** 5)
    return ghost


def test_fork_is_an_independent_copy_and_the_source_never_sees_the_executor():
    source = source_market([(1, "ask", 101, 5), (2, "bid", 99, 5)])
    ghost = fork(source)
    assert ghost.book.levels("ask") == source.book.levels("ask") == [(101, 5)]
    ghost.market(EXECUTOR, "bid", 3)
    assert ghost.book.levels("ask") == [(101, 2)]
    assert source.book.levels("ask") == [(101, 5)]             # the copy's fill never touches Arm 0
    assert source.accounts[1].shares == 10 ** 5 and ghost.accounts[1].shares == 10 ** 5 - 3
    assert EXECUTOR not in source.accounts


def test_r1_a_marketable_limit_lapses_at_its_limit_instead_of_resting():
    """In Arm 0 this order bought at 101. In the copy 101 is gone; its sender wanted to trade, not to quote."""
    source = source_market([(1, "ask", 101, 5), (1, "ask", 103, 5)])
    ghost = fork(source)
    ghost.market(EXECUTOR, "bid", 5)
    report = source.limit(2, "bid", 101, 5)
    assert report["filled_quantity"] == 5 and not report["resting_quantity"]
    assert ghost.book.levels("bid") == []                       # no quote appeared that no owner would ever cancel
    assert ghost.book.levels("ask") == [(103, 5)]               # and it did not pay more than its limit either
    assert ghost.stats["lapsed"] == 5


def test_r1_a_market_order_walks_the_copy_further_than_it_walked_arm_0():
    source = source_market([(1, "ask", 101, 5), (1, "ask", 102, 5)])
    ghost = fork(source)
    ghost.market(EXECUTOR, "bid", 5)
    source.market(2, "bid", 3)
    assert source.book.levels("ask") == [(101, 2), (102, 5)]
    assert ghost.book.levels("ask") == [(102, 2)]
    assert ghost.accounts[2].shares == 10 ** 5 + 3              # the replayed taker settled at the copy's prices
    assert ghost.accounts[2].cash == 10 ** 7 - 3 * 102


def test_r2_the_rested_part_of_an_order_rests_and_the_traded_part_takes():
    source = source_market([(1, "ask", 101, 5)])
    ghost = fork(source)
    report = source.limit(2, "bid", 101, 8)
    assert (report["filled_quantity"], report["resting_quantity"]) == (5, 3)
    assert ghost.book.levels("bid") == source.book.levels("bid") == [(101, 3)]
    assert ghost.book.levels("ask") == source.book.levels("ask") == []


def test_r2_a_passive_order_that_would_cross_in_the_copy_is_clamped_post_only():
    source = source_market([(1, "ask", 105, 5), (2, "bid", 95, 5)])
    ghost = fork(source)
    ghost.limit(EXECUTOR, "bid", 100, 3)                        # only the copy has a bid at 100
    report = source.limit(1, "ask", 99, 4)                      # passive in Arm 0, whose best bid is 95
    assert report["resting_quantity"] == 4
    assert source.book.best_ask == 99
    assert ghost.book.best_ask == 101 and ghost.book.best_bid == 100   # one tick behind, never crossed
    assert ghost.accounts[EXECUTOR].shares == 10 ** 5           # the clamp keeps it from filling the executor
    assert ghost.stats["clamped"] == 1


def test_r2_an_order_outside_the_copys_window_is_dropped_not_moved():
    source = source_market([(1, "ask", 105, 5), (2, "bid", 95, 5)])
    ghost = fork(source)
    ghost.limit(EXECUTOR, "ask", 96, 3)                         # the copy's ask window now ends at 96 + 19
    report = source.limit(1, "ask", 120, 4)
    assert report["resting_quantity"] == 4
    assert source.book.get_volume_at_price("ask", 120) == 4
    assert ghost.book.get_volume_at_price("ask", 120) == 0
    assert ghost.stats["dropped"] == 1
    source.cancel(1, report["order_id"])                        # and its later cancel finds nothing, harmlessly


def test_r3_a_cancel_removes_the_counterpart_or_is_a_no_op_if_the_executor_took_it():
    source = source_market()
    consumed = source.limit(1, "ask", 101, 5)["order_id"]
    resting = source.limit(1, "ask", 102, 5)["order_id"]
    ghost = fork(source)
    ghost.market(EXECUTOR, "bid", 5)
    source.cancel(1, consumed)
    source.cancel(1, resting)
    assert source.book.levels("ask") == [] and ghost.book.levels("ask") == []
    assert ghost.accounts[1].reserved_shares == 0


def test_r3_an_eviction_in_arm_0_removes_the_counterpart():
    source = source_market([(1, "bid", 80, 5), (1, "ask", 120, 5)])
    ghost = fork(source)
    report = source.limit(2, "bid", 100, 1)                     # shifts the bid window past 80
    assert [o["price"] for o in report["evicted_orders"]] == [80]
    assert ghost.book.get_volume_at_price("bid", 80) == 0
    assert ghost.accounts[1].reserved_cash == 0


@pytest.mark.parametrize("ttl", [3, None])
def test_r4_an_order_that_traded_in_arm_0_but_not_in_the_copy_expires_as_an_orphan(ttl):
    source = source_market()
    source.limit(1, "bid", 100, 5)
    source.limit(3, "ask", 110, 5)
    ghost = fork(source, ttl=ttl)
    ghost.limit(EXECUTOR, "bid", 101, 5)                        # the executor ahead of it in the copy
    source.market(2, "ask", 5)                                  # fills it in Arm 0, fills the executor in the copy
    assert source.book.get_volume_at_price("bid", 100) == 0
    assert ghost.accounts[EXECUTOR].shares == 10 ** 5 + 5
    assert ghost.book.get_volume_at_price("bid", 100) == 5      # conserved: the executor added this liquidity
    assert ghost.stats["orphaned"] == 1
    for _ in range(5):
        source.end_step()
        ghost.end_step()
    assert ghost.book.get_volume_at_price("bid", 100) == (0 if ttl == 3 else 5)
    assert ghost.stats["expired"] == (1 if ttl == 3 else 0)
    if ttl == 3:
        assert ghost.accounts[1].reserved_cash == 0


def test_r5_only_the_executor_is_budget_checked_and_the_copy_conserves():
    source = source_market([(1, "ask", 101, 5), (2, "bid", 99, 5)])
    ghost = GhostMarket(source)
    ghost.open_account(EXECUTOR, 0, 0)
    total = lambda m: (sum(a.cash for a in m.accounts.values()), sum(a.shares for a in m.accounts.values()))
    before = total(ghost)
    assert ghost.market(EXECUTOR, "bid", 5) is None             # no cash, no trade
    assert ghost.budget_rejects["bid"] == 1
    ghost.accounts[1].shares = 0                                # a shadow may run out: it was funded in Arm 0
    source.limit(1, "ask", 102, 5)
    assert ghost.book.get_volume_at_price("ask", 102) == 5
    source.market(2, "bid", 7)
    assert total(ghost)[0] == before[0]
    assert ghost.accounts[1].free_shares < 0


def test_book_ioc_never_rests_never_needs_the_window_and_accounts_for_every_lot():
    book = OrderBook(num_levels=5, mid_price=100, distance_from_mid=1)
    book.limit("ask", 1, 101, 3)
    book.limit("bid", 2, 99, 1)
    report = book.ioc("bid", 3, 150, 5)                         # 150 is far outside any window; it cannot rest anyway
    assert report["order_type"] == "ioc" and report["filled_quantity"] == 3 and report["resting_quantity"] == 0
    assert report["cancelled_quantity"] == 2
    assert_accounted(report)
    assert book.best_bid == 99 and book.best_ask is None
    assert book.client_order_ids(3) == {} or not book.client_order_ids(3)
    assert book.is_active(report["order_id"]) is False
    with pytest.raises(ValueError):
        book.ioc("nope", 3, 100, 1)


def test_book_invariants_hold_in_the_copy_with_a_trading_executor():
    from order_book.counterfactual import GHOST, PairedRun, PairedSpec, burned_in

    spec = PairedSpec(1000, "twap_passive", 0.4, GHOST, burn_in=300)
    blob, volume = burned_in(spec.seed, spec.config(), spec.burn_in)
    run = PairedRun(blob, GHOST)
    task = ExecutionTask(side="bid", size_fraction=0.4, horizon=150, decision_interval=5)
    executor = run.add_executor(task, make_strategy("twap_passive"), max(volume, 2.0))
    initial = sum(a.cash for a in run.market.accounts.values()), sum(a.shares for a in run.market.accounts.values())
    for _ in range(220):
        run.step()
        assert_book_invariants(run.market.book)
        assert_reservations_consistent(run.market, allow_short=True)
        assert (sum(a.cash for a in run.market.accounts.values()),
                sum(a.shares for a in run.market.accounts.values())) == initial
    assert executor.done and executor.result.executed == executor.quantity
    assert executor.account.free_cash >= 0 and executor.account.free_shares >= 0
    assert np.isfinite(executor.result.shortfall_bps)
