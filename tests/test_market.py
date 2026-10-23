"""Reservation bookkeeping must match the book exactly under every removal path."""

from order_book.book import OrderBook
from order_book.config import PublicStateConfig
from order_book.market import Market
from tests.helpers import assert_reservations_consistent


def test_settlement_reservations_and_removal_paths():
    book = OrderBook(5, 100, 1, self_trade_prevention="cancel_oldest")
    market = Market(book)
    for cid in range(3):
        market.open_account(cid, 10_000, 50)
    assert market.limit(0, "bid", 99, 10) is not None  # reserves 990
    assert market.accounts[0].reserved_cash == 990
    assert market.limit(1, "ask", 101, 5) is not None  # reserves 5 shares
    market.market(2, "ask", 4)  # hits client 0's bid: maker cash reservation released pro rata
    assert market.accounts[0].reserved_cash == 99 * 6
    assert (market.accounts[0].shares, market.accounts[2].cash) == (54, 10_000 + 4 * 99)
    market.limit(0, "bid", 95, 2)
    market.limit(0, "bid", 100, 1)  # improving the best evicts the 95 bid (window of 5)
    assert book.get_volume_at_price("bid", 95) == 0
    market.market(0, "ask", 3)  # own bid at best: cancel_oldest removes it and releases cash
    assert_reservations_consistent(market)
    market.cancel_all(0)
    market.cancel_all(1)
    assert (market.accounts[0].reserved_cash, market.accounts[1].reserved_shares) == (0, 0)
    assert market.limit(2, "ask", 200, 1) is None  # outside the window: dropped, counted
    assert market.window_rejects["ask"] == 1
    market.limit(1, "ask", 101, 50)
    assert market.accounts[1].reserved_shares == 50
    assert market.limit(1, "ask", 102, 1) is None  # nothing free left
    assert market.budget_rejects["ask"] == 1
    market.accounts[2].cash = 250
    report = market.market(2, "bid", 10)  # can afford only 2 at 101
    assert report["filled_quantity"] == 2
    assert_reservations_consistent(market)


def test_public_state_tracks_flow_window_and_imbalance():
    book = OrderBook(5, 100, 1)
    market = Market(book)
    market.open_account(0, 100_000, 100)
    market.open_account(1, 100_000, 100)
    market.limit(0, "ask", 101, 50)
    market.market(1, "bid", 10)
    market.end_step()
    ps = market.public
    assert (ps.buy_flow, ps.sell_flow, ps.imbalance) == (10, 0, 1.0)
    assert ps.last == 101
    for _ in range(PublicStateConfig().flow_window):  # the window rolls off completely
        market.end_step()
    assert (ps.buy_flow, ps.imbalance) == (0, 0.0)
