"""Deterministic, single-instrument price-time matching engine."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np


Side = Literal["bid", "ask"]


class OrderView(TypedDict):
    order_id: int
    client_id: int
    side: Side
    price: int
    quantity: int


class Trade(TypedDict):
    trade_id: int
    price: int
    quantity: int
    buyer_id: int
    seller_id: int
    buy_order_id: int
    sell_order_id: int
    aggressor_side: Literal["buy", "sell"]


class Report(TypedDict):
    order_id: int
    client_id: int
    side: Side
    order_type: str
    price: int | None
    requested_quantity: int
    filled_quantity: int
    resting_quantity: int
    cancelled_quantity: int
    avg_fill_price: float | None
    trades: list[Trade]
    stp_cancelled_order_ids: list[int]
    evicted_orders: list[OrderView]
    replaced_order_id: int | None


class Snapshot(TypedDict):
    best_bid: int | None
    best_ask: int | None
    mid_price: float | None
    spread: int | None
    bid_window: np.ndarray
    ask_window: np.ndarray
    bids: list[tuple[int, int, int]]
    asks: list[tuple[int, int, int]]
    total_bid_volume: int
    total_ask_volume: int


@dataclass
class _Order:
    order_id: int
    client_id: int
    side: Side
    price: int
    quantity: int

    def view(self) -> OrderView:
        return OrderView(
            order_id=self.order_id, client_id=self.client_id, side=self.side,
            price=self.price, quantity=self.quantity,
        )


class OrderBook:
    """FIFO matching with explicit bounded-window rejection and cancellation.

    Prices and quantities are integer ticks/lots, not floating-point currency.
    The book starts empty; constructor prices anchor admission/observation
    windows, not executable quotes. Calls must be serialized by the caller.
    """

    VALID_STP_MODES = frozenset({"none", "cancel_newest", "cancel_oldest", "cancel_both"})
    _MAX_INT = int(np.iinfo(np.int64).max)

    def __init__(
        self, num_levels: int, mid_price: int, distance_from_mid: int,
        self_trade_prevention: str = "cancel_newest",
    ):
        self.num_levels = self._integer(num_levels, "num_levels", minimum=1)
        self.initial_mid_price = self._integer(mid_price, "mid_price", minimum=1)
        self.distance_from_mid = self._integer(
            distance_from_mid, "distance_from_mid", minimum=1,
        )
        self._check_mode(self_trade_prevention)
        self.self_trade_prevention = self_trade_prevention
        bid = self.initial_mid_price - self.distance_from_mid
        ask = self.initial_mid_price + self.distance_from_mid
        if bid < 1 or ask > self._MAX_INT:
            raise ValueError("Initial window anchors must be positive int64 prices.")
        self._anchors: dict[Side, int] = {"bid": bid, "ask": ask}
        self._books: dict[Side, dict[int, OrderedDict[int, _Order]]] = {
            "bid": {}, "ask": {},
        }
        self._orders: dict[int, _Order] = {}
        self.order_id = 100_000
        self.trade_id = 1
        self._trades: list[Trade] = []
        self._last_report: Report | None = None

    @classmethod
    def _integer(cls, value: int, name: str, minimum: int = 0) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
        value = int(value)
        if not minimum <= value <= cls._MAX_INT:
            raise ValueError(f"{name} must be between {minimum} and {cls._MAX_INT}.")
        return value

    @classmethod
    def _check_mode(cls, mode: str) -> None:
        if mode not in cls.VALID_STP_MODES:
            raise ValueError(f"STP mode must be one of {sorted(cls.VALID_STP_MODES)}.")

    def _best(self, side: Side) -> int | None:
        book = self._books[side]
        return (max(book) if side == "bid" else min(book)) if book else None

    @property
    def best_bid(self) -> int | None:
        return self._best("bid")

    @property
    def best_ask(self) -> int | None:
        return self._best("ask")

    @property
    def mid_price(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid, self.best_ask
        return ask - bid if bid is not None and ask is not None else None

    def _window(self, side: Side) -> tuple[int, int]:
        anchor = self._anchors[side]
        edge = (max(1, anchor - self.num_levels + 1) if side == "bid"
                else min(self._MAX_INT, anchor + self.num_levels - 1))
        return anchor, edge

    @property
    def bid_prices(self) -> np.ndarray:
        return np.array(self._window("bid"), dtype=np.int64)

    @property
    def ask_prices(self) -> np.ndarray:
        return np.array(self._window("ask"), dtype=np.int64)

    def _table(self, side: Side) -> dict[int, list[OrderView]]:
        return {p: [o.view() for o in queue.values()]
                for p, queue in self._books[side].items()}

    @property
    def bid_table(self) -> dict[int, list[OrderView]]:
        return self._table("bid")

    @property
    def ask_table(self) -> dict[int, list[OrderView]]:
        return self._table("ask")

    @property
    def trades(self) -> list[Trade]:
        return deepcopy(self._trades)

    @property
    def last_order_report(self) -> Report | None:
        return deepcopy(self._last_report)

    def _validate_price(self, side: Side, price: int) -> int:
        price = self._integer(price, "price", minimum=1)
        edge = self._window(side)[1]
        if (side == "bid" and price < edge) or (side == "ask" and price > edge):
            raise ValueError(f"{side} price {price} is outside the active window.")
        return price

    def _remove(self, order: _Order) -> None:
        queue = self._books[order.side][order.price]
        del queue[order.order_id]
        del self._orders[order.order_id]
        if not queue:
            del self._books[order.side][order.price]

    def _refresh(self, side: Side) -> list[OrderView]:
        best = self._best(side)
        if best is None:
            return []
        self._anchors[side] = best
        edge = self._window(side)[1]
        evicted = []
        for price, queue in list(self._books[side].items()):
            if (side == "bid" and price < edge) or (side == "ask" and price > edge):
                for order in list(queue.values()):
                    evicted.append(order.view())
                    self._remove(order)
        return evicted

    def _match(
        self, incoming: _Order, limit: int | None, mode: str,
    ) -> tuple[list[Trade], list[int], int]:
        opposite: Side = "ask" if incoming.side == "bid" else "bid"
        book = self._books[opposite]
        trades: list[Trade] = []
        cancelled_ids: list[int] = []
        cancelled = 0
        for price in sorted(book, reverse=opposite == "bid"):
            if limit is not None and (
                price > limit if incoming.side == "bid" else price < limit
            ):
                break
            queue = book[price]
            while queue and incoming.quantity:
                resting = next(iter(queue.values()))
                if mode != "none" and incoming.client_id == resting.client_id:
                    if mode in {"cancel_oldest", "cancel_both"}:
                        cancelled_ids.append(resting.order_id)
                        self._remove(resting)
                    if mode in {"cancel_newest", "cancel_both"}:
                        cancelled = incoming.quantity
                        incoming.quantity = 0
                    continue
                quantity = min(incoming.quantity, resting.quantity)
                buyer, seller = ((incoming, resting) if incoming.side == "bid"
                                 else (resting, incoming))
                trade = Trade(
                    trade_id=self.trade_id, price=price, quantity=quantity,
                    buyer_id=buyer.client_id, seller_id=seller.client_id,
                    buy_order_id=buyer.order_id, sell_order_id=seller.order_id,
                    aggressor_side="buy" if incoming.side == "bid" else "sell",
                )
                self.trade_id += 1
                self._trades.append(trade)
                trades.append(trade)
                incoming.quantity -= quantity
                resting.quantity -= quantity
                if not resting.quantity:
                    self._remove(resting)
            if not incoming.quantity:
                break
        self._refresh(opposite)
        return trades, cancelled_ids, cancelled

    def _submit(
        self, side: Side, client_id: int, quantity: int, price: int | None,
        stp_mode: str | None, replacement: _Order | None = None,
    ) -> Report:
        client_id = self._integer(client_id, "client_id")
        quantity = self._integer(quantity, "quantity", minimum=1)
        if price is not None:
            price = self._validate_price(side, price)
        mode = self.self_trade_prevention if stp_mode is None else stp_mode
        self._check_mode(mode)
        # Validate the entire replacement before changing the original queue.
        if replacement is not None:
            self._remove(replacement)
        order = _Order(self.order_id, client_id, side, price or 0, quantity)
        self.order_id += 1
        trades, stp_ids, cancelled = self._match(order, price, mode)
        if price is None:
            cancelled += order.quantity
            order.quantity = 0
        elif order.quantity:
            self._books[side].setdefault(price, OrderedDict())[order.order_id] = order
            self._orders[order.order_id] = order
        evicted = self._refresh(side)
        filled = sum(t["quantity"] for t in trades)
        report = Report(
            order_id=order.order_id, client_id=client_id, side=side,
            order_type="market" if price is None else "limit", price=price,
            requested_quantity=quantity, filled_quantity=filled,
            resting_quantity=order.quantity, cancelled_quantity=cancelled,
            avg_fill_price=(sum(t["price"] * t["quantity"] for t in trades) / filled
                            if filled else None),
            trades=trades, stp_cancelled_order_ids=stp_ids, evicted_orders=evicted,
            replaced_order_id=replacement.order_id if replacement else None,
        )
        self._last_report = report
        return deepcopy(report)

    def buy(self, client_id: int, quantity: int = 1, stp_mode: str | None = None) -> Report:
        return self._submit("bid", client_id, quantity, None, stp_mode)

    def sell(self, client_id: int, quantity: int = 1, stp_mode: str | None = None) -> Report:
        return self._submit("ask", client_id, quantity, None, stp_mode)

    def place_bid(
        self, client_id: int, bid: int, quantity: int, stp_mode: str | None = None,
    ) -> int:
        return self._submit(
            "bid", client_id, quantity, self._integer(bid, "bid", minimum=1), stp_mode,
        )["order_id"]

    def place_ask(
        self, client_id: int, ask: int, quantity: int, stp_mode: str | None = None,
    ) -> int:
        return self._submit(
            "ask", client_id, quantity, self._integer(ask, "ask", minimum=1), stp_mode,
        )["order_id"]

    def _owned(self, order_id: int, client_id: int | None) -> _Order:
        order_id = self._integer(order_id, "order_id")
        if client_id is not None:
            client_id = self._integer(client_id, "client_id")
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} is not active.")
        if client_id is not None and order.client_id != client_id:
            raise PermissionError(f"Client {client_id} does not own order {order_id}.")
        return order

    def _lifecycle_report(
        self, order: _Order, operation: str, cancelled: int,
    ) -> Report:
        report = Report(
            order_id=order.order_id, client_id=order.client_id, side=order.side,
            order_type=operation, price=order.price,
            requested_quantity=order.quantity + cancelled, filled_quantity=0,
            resting_quantity=order.quantity, cancelled_quantity=cancelled,
            avg_fill_price=None, trades=[], stp_cancelled_order_ids=[],
            evicted_orders=[], replaced_order_id=None,
        )
        self._last_report = report
        return deepcopy(report)

    def cancel_order(self, order_id: int, client_id: int | None = None) -> Report:
        order = self._owned(order_id, client_id)
        self._remove(order)
        cancelled, order.quantity = order.quantity, 0
        self._refresh(order.side)
        return self._lifecycle_report(order, "cancel", cancelled)

    def modify_order(
        self, order_id: int, new_quantity: int | None = None,
        new_price: int | None = None, client_id: int | None = None,
    ) -> int | None:
        """Quantity is the desired remaining size, not the original submitted size."""
        order = self._owned(order_id, client_id)
        quantity = (order.quantity if new_quantity is None
                    else self._integer(new_quantity, "new_quantity"))
        price = order.price if new_price is None else self._validate_price(order.side, new_price)
        if quantity == 0:
            self.cancel_order(order_id, client_id)
            return None
        if price == order.price and quantity <= order.quantity:
            cancelled = order.quantity - quantity
            order.quantity = quantity
            self._lifecycle_report(order, "modify", cancelled)
            return order_id
        return self._submit(
            order.side, order.client_id, quantity, price, None, replacement=order,
        )["order_id"]

    def get_order(self, order_id: int) -> OrderView | None:
        order_id = self._integer(order_id, "order_id")
        order = self._orders.get(order_id)
        return order.view() if order else None

    def get_client_orders(self, client_id: int) -> list[OrderView]:
        client_id = self._integer(client_id, "client_id")
        return [o.view() for o in self._orders.values() if o.client_id == client_id]

    def get_volume_at_price(self, side: Side, price: int) -> int:
        if side not in self._books:
            raise ValueError("side must be 'bid' or 'ask'.")
        price = self._integer(price, "price", minimum=1)
        return sum(o.quantity for o in self._books[side].get(price, {}).values())

    def _depth(self, depth: int | None) -> int:
        return self.num_levels if depth is None else self._integer(depth, "depth")

    def get_snapshot(self, depth: int | None = None) -> Snapshot:
        """L2 populated levels; total volumes always include the whole active book."""
        depth = self._depth(depth)
        levels = {
            side: [(p, sum(o.quantity for o in book[p].values()), len(book[p]))
                   for p in sorted(book, reverse=side == "bid")[:depth]]
            for side, book in self._books.items()
        }
        return {
            "best_bid": self.best_bid, "best_ask": self.best_ask,
            "mid_price": self.mid_price, "spread": self.spread,
            "bid_window": self.bid_prices, "ask_window": self.ask_prices,
            "bids": levels["bid"], "asks": levels["ask"],
            "total_bid_volume": sum(o.quantity for o in self._orders.values() if o.side == "bid"),
            "total_ask_volume": sum(o.quantity for o in self._orders.values() if o.side == "ask"),
        }

    def get_book_arrays(self, depth: int | None = None) -> dict[str, np.ndarray]:
        """Contiguous tick grids; invalid nonpositive/overflow prices are masked out."""
        depth = self._depth(depth)
        arrays = {}
        sides: tuple[Side, Side] = ("bid", "ask")
        for side in sides:
            anchor = self._anchors[side]
            prices = [anchor + i * (-1 if side == "bid" else 1) for i in range(depth)]
            valid = np.array([1 <= p <= self._MAX_INT for p in prices], dtype=bool)
            arrays[f"{side}_prices"] = np.array(
                [p if ok else 0 for p, ok in zip(prices, valid)], dtype=np.int64,
            )
            volumes = [self.get_volume_at_price(side, p) if ok else 0
                       for p, ok in zip(prices, valid)]
            # Aggregate volume can exceed int64 even when each order is valid.
            if any(v > self._MAX_INT for v in volumes):
                raise OverflowError("Aggregated depth volume exceeds int64.")
            arrays[f"{side}_volumes"] = np.array(volumes, dtype=np.int64)
            arrays[f"{side}_valid"] = valid
        return arrays
