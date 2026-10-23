"""Invariant checks shared by the test modules."""

from order_book.book import OrderBook
from order_book.market import Market


def assert_accounted(report) -> None:
    assert report["requested_quantity"] == (
        report["filled_quantity"] + report["resting_quantity"] + report["cancelled_quantity"]
    )


def assert_book_invariants(book: OrderBook) -> None:
    """Tables, indexes, per-level volumes and best quotes all agree; the book is never crossed."""
    orders = []
    for side, table in (("bid", book.bid_table), ("ask", book.ask_table)):
        window = book.bid_prices if side == "bid" else book.ask_prices
        for price, queue in table.items():
            assert queue
            assert min(window) <= price <= max(window)
            ids = [o["order_id"] for o in queue]
            assert ids == sorted(ids)
            for order in queue:
                assert order["quantity"] > 0
                assert book.get_order(order["order_id"]) == order
                orders.append(order)
    assert len(orders) == len(book._orders)
    assert {o["order_id"] for o in orders} == set(book._orders)
    for side in ("bid", "ask"):
        table = book.bid_table if side == "bid" else book.ask_table
        assert sorted(table) == book._prices[side]
        for price, queue in table.items():
            assert book.get_volume_at_price(side, price) == sum(o["quantity"] for o in queue)
    by_client: dict[int, set[int]] = {}
    for o in orders:
        by_client.setdefault(o["client_id"], set()).add(o["order_id"])
    for cid, ids in by_client.items():
        assert {o["order_id"] for o in book.get_client_orders(cid)} == ids
    if book.best_bid is not None and book.best_ask is not None:
        assert book.best_bid < book.best_ask
        assert book.mid_price == (book.best_bid + book.best_ask) / 2
    else:
        assert book.mid_price is None
        assert book.spread is None
    assert_accounted(book.last_order_report)


def assert_reservations_consistent(market: Market) -> None:
    """Running reservation totals equal what actually rests; nobody is short of cash or shares."""
    reserved = {cid: [0, 0] for cid in market.accounts}
    for side, table in (("bid", market.book.bid_table), ("ask", market.book.ask_table)):
        for queue in table.values():
            for o in queue:
                reserved[o["client_id"]][side == "ask"] += o["price"] * o["quantity"] if side == "bid" else o["quantity"]
    for cid, account in market.accounts.items():
        assert (account.reserved_cash, account.reserved_shares) == tuple(reserved[cid]), cid
        assert account.free_cash >= 0
        assert account.free_shares >= 0
