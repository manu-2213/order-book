import random

import numpy as np
import pytest

from order_book.book import OrderBook
from tests.helpers import assert_book_invariants


@pytest.fixture
def book():
    return OrderBook(20, 10000, 5)


def test_empty_and_one_sided_state(book):
    assert book.mid_price is None
    assert book.best_bid is None
    assert book.best_ask is None
    np.testing.assert_array_equal(book.bid_prices, [9995, 9976])
    np.testing.assert_array_equal(book.ask_prices, [10005, 10024])
    bid = book.place_bid(1, 9990, 5)
    assert book.bid_prices[0] == 9990
    assert book.mid_price is None
    book.place_ask(2, 10010, 4)
    assert book.ask_prices[0] == 10010
    assert book.mid_price == 10000
    book.cancel_order(bid, 1)
    assert book.mid_price is None
    assert book.spread is None


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_exact_window_boundaries_and_reported_evictions(book, side):
    place = book.place_bid if side == "bid" else book.place_ask
    edge, outside, improved = (9976, 9975, 9998) if side == "bid" else (10024, 10025, 10002)
    with pytest.raises(ValueError):
        place(1, outside, 1)
    old = place(1, edge, 7)
    place(2, improved, 3)
    assert book.get_order(old) is None
    assert book.last_order_report["evicted_orders"][0]["order_id"] == old
    assert book.last_order_report["evicted_orders"][0]["quantity"] == 7
    assert_book_invariants(book)


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_all_admitted_window_ticks_appear_in_arrays(book, side):
    place = book.place_bid if side == "bid" else book.place_ask
    anchor = 9995 if side == "bid" else 10005
    step = -1 if side == "bid" else 1
    for i in range(20):
        place(i, anchor + i * step, i + 1)
    arrays = book.get_book_arrays()
    assert arrays[f"{side}_volumes"].tolist() == list(range(1, 21))
    assert arrays[f"{side}_prices"].tolist() == [anchor + i * step for i in range(20)]
    assert_book_invariants(book)


def test_empty_side_window_follows_opposite_best_without_fabricating_quote(book):
    first = book.place_ask(1, 10010, 2)
    book.place_ask(2, 10015, 4)
    np.testing.assert_array_equal(book.bid_prices, [10009, 9990])
    book.buy(3, 2)
    assert book.get_order(first) is None
    assert book.ask_prices[0] == 10015
    np.testing.assert_array_equal(book.bid_prices, [10014, 9995])
    assert book.mid_price is None
    with pytest.raises(ValueError):
        book.place_bid(3, 9994, 1)
    remaining = book.get_client_orders(2)[0]["order_id"]
    book.cancel_order(remaining, 2)
    assert book.best_ask is None
    # Both sides empty: both windows anchor on the last trade price.
    assert book.ask_prices[0] == 10010
    assert book.bid_prices[0] == 10010
    assert OrderBook(20, 10000, 5).ask_prices[0] == 10005  # no trade yet


def test_default_stp_market_cancels_incoming_not_resting(book):
    order_id = book.place_ask(1, 10005, 5)
    report = book.buy(1, 3)
    assert report["filled_quantity"] == 0
    assert report["cancelled_quantity"] == 3
    assert book.get_order(order_id)["quantity"] == 5
    assert_book_invariants(book)


@pytest.mark.parametrize("buying", [True, False])
def test_price_priority_fifo_partial_and_market_ioc(book, buying):
    place = book.place_ask if buying else book.place_bid
    market = book.buy if buying else book.sell
    best, worse = (10005, 10008) if buying else (9995, 9992)
    deep = place(1, worse, 4)
    first = place(2, best, 3)
    second = place(3, best, 5)
    report = market(9, 6)
    key = "sell_order_id" if buying else "buy_order_id"
    assert [t[key] for t in report["trades"]] == [first, second]
    assert [t["quantity"] for t in report["trades"]] == [3, 3]
    assert report["avg_fill_price"] == best
    assert book.get_order(first) is None
    assert book.get_order(second)["quantity"] == 2
    report = market(9, 10)
    assert [t[key] for t in report["trades"]] == [second, deep]
    assert report["filled_quantity"] == 6
    assert report["cancelled_quantity"] == 4
    assert report["avg_fill_price"] == pytest.approx((2 * best + 4 * worse) / 6)
    assert report["trades"][0]["buyer_id"] == (9 if buying else 3)
    assert report["trades"][0]["seller_id"] == (3 if buying else 9)
    assert market(9, 2)["cancelled_quantity"] == 2
    assert_book_invariants(book)


@pytest.mark.parametrize("buying", [True, False])
def test_crossing_limits_keep_price_protection_and_rest(book, buying):
    maker = book.place_ask if buying else book.place_bid
    taker = book.place_bid if buying else book.place_ask
    prices = (10005, 10007, 10010) if buying else (9995, 9993, 9990)
    for client, price in enumerate(prices):
        maker(client, price, 10)
    limit = 10008 if buying else 9992
    order_id = taker(9, limit, 25)
    report = book.last_order_report
    assert [t["price"] for t in report["trades"]] == list(prices[:2])
    assert report["filled_quantity"] == 20
    assert book.get_order(order_id)["quantity"] == 5
    assert book.get_order(order_id)["price"] == limit
    assert_book_invariants(book)


@pytest.mark.parametrize("buying", [True, False])
@pytest.mark.parametrize("mode", sorted(OrderBook.VALID_STP_MODES))
def test_all_stp_modes_after_external_partial_fill(buying, mode):
    book = OrderBook(20, 10000, 5, mode)
    maker = book.place_ask if buying else book.place_bid
    taker = book.place_bid if buying else book.place_ask
    price = 10005 if buying else 9995
    maker(2, price, 2)
    own = maker(1, price, 3)
    maker(3, price, 4)
    taker(1, price, 8)
    report = book.last_order_report
    expected = {
        "none": (8, 0, 0), "cancel_newest": (2, 0, 6), "cancel_oldest": (6, 2, 0), "cancel_both": (2, 0, 6),
    }[mode]
    assert tuple(report[k] for k in ("filled_quantity", "resting_quantity", "cancelled_quantity")) == expected
    assert [o["order_id"] for o in report["stp_cancelled_orders"]] == (
        [own] if mode in {"cancel_oldest", "cancel_both"} else []
    )
    assert (book.get_order(own) is not None) == (mode == "cancel_newest")
    if mode != "none":
        assert all(t["buyer_id"] != t["seller_id"] for t in book.trades)
    assert_book_invariants(book)


def test_cancel_modify_priority_and_crossing(book):
    first = book.place_bid(1, 9995, 10)
    second = book.place_bid(2, 9995, 4)
    assert book.modify_order(first, 5, client_id=1) == first
    assert book.last_order_report["cancelled_quantity"] == 5
    increased = book.modify_order(first, 8, client_id=1)
    assert [o["order_id"] for o in book.bid_table[9995]] == [second, increased]
    assert book.last_order_report["replaced_order_id"] == first
    book.place_ask(3, 10005, 3)
    crossed = book.modify_order(increased, new_price=10005, client_id=1)
    assert book.last_order_report["filled_quantity"] == 3
    assert book.get_order(crossed)["quantity"] == 5
    assert book.modify_order(crossed, 0, client_id=1) is None
    assert book.get_order(crossed) is None
    assert_book_invariants(book)


@pytest.mark.parametrize("side", ["bid", "ask"])
def test_rejected_modifications_are_atomic(side):
    book = OrderBook(20, 10000, 5)
    place = book.place_bid if side == "bid" else book.place_ask
    order_id = place(1, 9995 if side == "bid" else 10005, 10)
    before, next_id = book.get_order(order_id), book.order_id
    changes = [
        {"new_quantity": -1}, {"new_quantity": 1.9}, {"new_quantity": True},
        {"new_price": 9990.5}, {"new_price": 0},
        {"new_price": 9975 if side == "bid" else 10025},
    ]
    for change in changes:
        with pytest.raises(ValueError):
            book.modify_order(order_id, client_id=1, **change)
        assert book.get_order(order_id) == before
        assert book.order_id == next_id
    for operation in (book.cancel_order, book.modify_order):
        with pytest.raises(PermissionError):
            operation(order_id, client_id=2)
        with pytest.raises(KeyError):
            operation(0, client_id=1)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", 2**63])
def test_invalid_inputs(book, value):
    with pytest.raises(ValueError):
        OrderBook(value, 10000, 5)
    with pytest.raises(ValueError):
        book.buy(1, value)
    with pytest.raises(ValueError):
        book.place_bid(1, value, 1)


def test_invalid_constructor_price_none_and_stp_mode_and_numpy_integers(book):
    with pytest.raises(ValueError):
        OrderBook(20, 5, 5)
    with pytest.raises(ValueError):
        book.place_bid(1, None, 1)
    with pytest.raises(ValueError):
        book.buy(1, 1, "invalid")
    order_id = book.place_bid(np.int64(1), np.int64(9995), np.int64(3))
    assert book.get_order(order_id)["quantity"] == 3


def test_snapshots_arrays_depth_and_mutation_isolation(book):
    bid = book.place_bid(1, 9995, 5)
    book.place_bid(2, 9993, 7)
    ask = book.place_ask(3, 10005, 4)
    snapshot = book.get_snapshot(1)
    assert snapshot["bids"] == [(9995, 5, 1)]
    assert snapshot["total_bid_volume"] == 12
    assert book.get_snapshot(0)["bids"] == []
    arrays = book.get_book_arrays()
    assert arrays["bid_prices"].shape == (20,)
    assert arrays["bid_prices"].dtype == np.int64
    assert arrays["bid_volumes"][:3].tolist() == [5, 0, 7]
    assert book.get_book_arrays(0)["ask_prices"].size == 0
    for invalid in (-1, 2.5, True):
        for query in (book.get_snapshot, book.get_book_arrays):
            with pytest.raises(ValueError):
                query(invalid)
    book.bid_table[9995][0]["quantity"] = 0
    book.get_order(bid)["quantity"] = 0
    assert book.get_order(bid)["quantity"] == 5
    # Reports and trade records are shared, read-only objects (no deep copies).
    result = book.buy(4, 2)
    assert result is book.last_order_report
    assert result["trades"][0]["price"] == 10005
    assert result["trades"][0]["sell_order_id"] == ask
    book.trades.clear()
    assert len(book.trades) == 1
    assert len(OrderBook(20, 10000, 5, keep_trades=False).trades) == 0


def test_low_prices_and_aggregate_volume_overflow():
    book = OrderBook(5, 3, 1)
    book.place_bid(1, 1, 1)
    arrays = book.get_book_arrays()
    assert arrays["bid_prices"].tolist() == [1, 0, 0, 0, 0]
    assert arrays["bid_valid"].tolist() == [True, False, False, False, False]
    book.place_ask(2, 4, 2**63 - 1)
    book.place_ask(3, 4, 1)
    with pytest.raises(OverflowError):
        book.get_book_arrays()
    assert book.get_snapshot()["total_ask_volume"] == 2**63


def test_deterministic_random_sequences_preserve_invariants(book):
    rng = random.Random(42)
    for _ in range(600):
        action, client, quantity = rng.randrange(6), rng.randrange(5), rng.randrange(1, 20)
        if action < 2:
            side = "bid" if action == 0 else "ask"
            anchor = int((book.bid_prices if side == "bid" else book.ask_prices)[0])
            price = anchor + rng.randrange(0, 5) * (1 if side == "bid" else -1)
            place = book.place_bid if side == "bid" else book.place_ask
            place(client, price, quantity)
        elif action < 4:
            market = book.buy if action == 2 else book.sell
            market(client, quantity)
        elif book._orders:
            order = rng.choice(list(book._orders.values()))
            if action == 4:
                book.cancel_order(order.order_id, order.client_id)
            else:
                book.modify_order(order.order_id, quantity, client_id=order.client_id)
        else:
            continue
        assert_book_invariants(book)
