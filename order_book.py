"""Deterministic single-instrument price-time matching engine.

Prices and quantities are integer ticks/lots. The book only matches, rests,
cancels and evicts orders; it assumes nothing about who trades or why. Best
prices and per-level volumes are maintained incrementally, a per-client index
makes order lookup O(1), and every report names the maker order of each fill
and every order removed by a window shift, so an external accountant can keep
reservations exact without scanning the book. Calls must be serialized.
"""

from bisect import bisect_left, insort
from collections import OrderedDict
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
    stp_cancelled_orders: list[OrderView]
    evicted_orders: list[OrderView]
    replaced_order_id: int | None


class _Order:
    __slots__ = ("order_id", "client_id", "side", "price", "quantity")

    def __init__(self, order_id: int, client_id: int, side: Side, price: int, quantity: int):
        self.order_id, self.client_id, self.side = order_id, client_id, side
        self.price, self.quantity = price, quantity

    def view(self) -> OrderView:
        return {"order_id": self.order_id, "client_id": self.client_id, "side": self.side,
                "price": self.price, "quantity": self.quantity}


class OrderBook:
    """FIFO matching with a bounded price window per side.

    Each side admits prices within ``num_levels`` ticks of its anchor (see
    ``anchor``); the constructor prices only seed the anchors of the empty
    book. Improving a side's best quote can evict that side's deepest orders;
    they are reported.
    """

    VALID_STP_MODES = frozenset({"none", "cancel_newest", "cancel_oldest", "cancel_both"})
    _MAX_INT = int(np.iinfo(np.int64).max)

    def __init__(self, num_levels: int, mid_price: int, distance_from_mid: int,
                 self_trade_prevention: str = "cancel_newest", keep_trades: bool = True):
        self.num_levels = self._integer(num_levels, "num_levels", 1)
        self.initial_mid_price = self._integer(mid_price, "mid_price", 1)
        self.distance_from_mid = self._integer(distance_from_mid, "distance_from_mid", 1)
        self._check_mode(self_trade_prevention)
        self.self_trade_prevention = self_trade_prevention
        self.keep_trades = keep_trades
        bid, ask = mid_price - distance_from_mid, mid_price + distance_from_mid
        if bid < 1 or ask > self._MAX_INT:
            raise ValueError("Initial window anchors must be positive int64 prices.")
        self._anchors: dict[Side, int] = {"bid": bid, "ask": ask}
        self._queues: dict[Side, dict[int, OrderedDict[int, _Order]]] = {"bid": {}, "ask": {}}
        self._volumes: dict[Side, dict[int, int]] = {"bid": {}, "ask": {}}
        self._prices: dict[Side, list[int]] = {"bid": [], "ask": []}  # ascending
        self._orders: dict[int, _Order] = {}
        self._client_orders: dict[int, dict[int, None]] = {}  # insertion-ordered id sets
        self.order_id = 100_000
        self.trade_id = 1
        self.last_trade_price: int | None = None
        self._trades: list[Trade] = []
        self._last_report: Report | None = None

    # ----- validation ------------------------------------------------------ #

    @classmethod
    def _integer(cls, value, name: str, minimum: int = 0) -> int:
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

    def anchor(self, side: Side) -> int:
        """Window anchor: the side's best price; if the side is empty, one tick
        inside the opposite best; if both are empty, the last trade price
        (or the constructor anchor before any trade)."""
        prices = self._prices[side]
        if prices:
            return prices[-1] if side == "bid" else prices[0]
        other = self._prices["ask" if side == "bid" else "bid"]
        if other:
            return other[0] - 1 if side == "bid" else other[-1] + 1
        return self.last_trade_price if self.last_trade_price is not None else self._anchors[side]

    def edge(self, side: Side) -> int:
        anchor = self.anchor(side)
        return (max(1, anchor - self.num_levels + 1) if side == "bid"
                else min(self._MAX_INT, anchor + self.num_levels - 1))

    def _validate_price(self, side: Side, price) -> int:
        price = self._integer(price, "price", 1)
        edge = self.edge(side)
        if (side == "bid" and price < edge) or (side == "ask" and price > edge):
            raise ValueError(f"{side} price {price} is outside the active window.")
        return price

    # ----- O(1) state ------------------------------------------------------ #

    @property
    def best_bid(self) -> int | None:
        prices = self._prices["bid"]
        return prices[-1] if prices else None

    @property
    def best_ask(self) -> int | None:
        prices = self._prices["ask"]
        return prices[0] if prices else None

    @property
    def mid_price(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid, self.best_ask
        return ask - bid if bid is not None and ask is not None else None

    @property
    def bid_prices(self) -> np.ndarray:
        return np.array([self.anchor("bid"), self.edge("bid")], dtype=np.int64)

    @property
    def ask_prices(self) -> np.ndarray:
        return np.array([self.anchor("ask"), self.edge("ask")], dtype=np.int64)

    def levels(self, side: Side) -> list[tuple[int, int]]:
        """(price, volume) pairs, best first."""
        prices = self._prices[side]
        volumes = self._volumes[side]
        return [(p, volumes[p]) for p in (reversed(prices) if side == "bid" else prices)]

    def best_excluding(self, side: Side, order_id: int) -> int | None:
        """Best price on ``side`` once ``order_id`` is ignored (everyone else's best quote)."""
        prices, queues = self._prices[side], self._queues[side]
        for price in (reversed(prices) if side == "bid" else prices):
            queue = queues[price]
            if len(queue) > 1 or order_id not in queue:
                return price
        return None

    def get_volume_at_price(self, side: Side, price: int) -> int:
        if side not in self._queues:
            raise ValueError("side must be 'bid' or 'ask'.")
        return self._volumes[side].get(self._integer(price, "price", 1), 0)

    def get_order(self, order_id: int) -> OrderView | None:
        order = self._orders.get(self._integer(order_id, "order_id"))
        return order.view() if order else None

    def get_client_orders(self, client_id: int) -> list[OrderView]:
        ids = self._client_orders.get(self._integer(client_id, "client_id"), ())
        return [self._orders[i].view() for i in ids]

    def client_order_ids(self, client_id: int):
        """Live, insertion-ordered view of a client's order ids; do not mutate."""
        return self._client_orders.get(client_id, ())

    @property
    def bid_table(self) -> dict[int, list[OrderView]]:
        return {p: [o.view() for o in q.values()] for p, q in self._queues["bid"].items()}

    @property
    def ask_table(self) -> dict[int, list[OrderView]]:
        return {p: [o.view() for o in q.values()] for p, q in self._queues["ask"].items()}

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    @property
    def last_order_report(self) -> Report | None:
        return self._last_report

    # ----- mutation primitives --------------------------------------------- #

    def _add(self, order: _Order) -> None:
        side, price = order.side, order.price
        queue = self._queues[side].get(price)
        if queue is None:
            queue = self._queues[side][price] = OrderedDict()
            self._volumes[side][price] = 0
            insort(self._prices[side], price)
        queue[order.order_id] = order
        self._volumes[side][price] += order.quantity
        self._orders[order.order_id] = order
        self._client_orders.setdefault(order.client_id, {})[order.order_id] = None

    def _remove(self, order: _Order) -> None:
        side, price = order.side, order.price
        queue = self._queues[side][price]
        del queue[order.order_id]
        del self._orders[order.order_id]
        del self._client_orders[order.client_id][order.order_id]
        if queue:
            self._volumes[side][price] -= order.quantity
        else:
            del self._queues[side][price], self._volumes[side][price]
            prices = self._prices[side]
            del prices[bisect_left(prices, price)]

    def _refresh(self, side: Side) -> list[OrderView]:
        """Evict orders that fell outside the window after the best price moved."""
        prices = self._prices[side]
        if not prices:
            return []
        edge = self.edge(side)
        evicted: list[OrderView] = []
        while prices and (prices[0] < edge if side == "bid" else prices[-1] > edge):
            for order in list(self._queues[side][prices[0] if side == "bid" else prices[-1]].values()):
                evicted.append(order.view())
                self._remove(order)
        return evicted

    def _match(self, incoming: _Order, limit: int | None, mode: str):
        side = incoming.side
        opposite: Side = "ask" if side == "bid" else "bid"
        prices, queues, volumes = self._prices[opposite], self._queues[opposite], self._volumes[opposite]
        trades: list[Trade] = []
        stp_cancelled: list[OrderView] = []
        cancelled = 0
        while prices and incoming.quantity:
            price = prices[0] if side == "bid" else prices[-1]
            if limit is not None and (price > limit if side == "bid" else price < limit):
                break
            resting = next(iter(queues[price].values()))
            if mode != "none" and resting.client_id == incoming.client_id:
                if mode != "cancel_newest":
                    stp_cancelled.append(resting.view())
                    self._remove(resting)
                if mode != "cancel_oldest":
                    cancelled, incoming.quantity = incoming.quantity, 0
                continue
            quantity = min(incoming.quantity, resting.quantity)
            buyer, seller = (incoming, resting) if side == "bid" else (resting, incoming)
            trade: Trade = {
                "trade_id": self.trade_id, "price": price, "quantity": quantity,
                "buyer_id": buyer.client_id, "seller_id": seller.client_id,
                "buy_order_id": buyer.order_id, "sell_order_id": seller.order_id,
                "aggressor_side": "buy" if side == "bid" else "sell",
            }
            self.trade_id += 1
            self.last_trade_price = price
            trades.append(trade)
            if self.keep_trades:
                self._trades.append(trade)
            incoming.quantity -= quantity
            if quantity == resting.quantity:
                self._remove(resting)
            else:
                resting.quantity -= quantity
                volumes[price] -= quantity
        return trades, stp_cancelled, cancelled, self._refresh(opposite)

    def _submit(self, side: Side, client_id, quantity, price, stp_mode,
                replacement: _Order | None = None) -> Report:
        client_id = self._integer(client_id, "client_id")
        quantity = self._integer(quantity, "quantity", 1)
        if price is not None:
            price = self._validate_price(side, price)
        mode = self.self_trade_prevention if stp_mode is None else stp_mode
        self._check_mode(mode)
        if replacement is not None:  # validated above, so the swap is atomic
            self._remove(replacement)
        order = _Order(self.order_id, client_id, side, price or 0, quantity)
        self.order_id += 1
        trades, stp_cancelled, cancelled, evicted = self._match(order, price, mode)
        if price is None:
            cancelled += order.quantity
            order.quantity = 0
        elif order.quantity:
            self._add(order)
        evicted += self._refresh(side)
        filled = quantity - order.quantity - cancelled
        report: Report = {
            "order_id": order.order_id, "client_id": client_id, "side": side,
            "order_type": "market" if price is None else "limit", "price": price,
            "requested_quantity": quantity, "filled_quantity": filled,
            "resting_quantity": order.quantity, "cancelled_quantity": cancelled,
            "avg_fill_price": (sum(t["price"] * t["quantity"] for t in trades) / filled
                               if filled else None),
            "trades": trades, "stp_cancelled_orders": stp_cancelled,
            "evicted_orders": evicted,
            "replaced_order_id": replacement.order_id if replacement else None,
        }
        self._last_report = report
        return report

    # ----- public API ------------------------------------------------------ #

    def limit(self, side: Side, client_id: int, price: int, quantity: int,
              stp_mode: str | None = None) -> Report:
        """GTC limit order: match up to ``price``, rest the remainder."""
        if side not in self._queues:
            raise ValueError("side must be 'bid' or 'ask'.")
        return self._submit(side, client_id, quantity, self._integer(price, "price", 1), stp_mode)

    def market(self, side: Side, client_id: int, quantity: int,
               stp_mode: str | None = None) -> Report:
        """Immediate-or-cancel market order; whatever cannot fill is cancelled."""
        if side not in self._queues:
            raise ValueError("side must be 'bid' or 'ask'.")
        return self._submit(side, client_id, quantity, None, stp_mode)

    def buy(self, client_id: int, quantity: int = 1, stp_mode: str | None = None) -> Report:
        return self.market("bid", client_id, quantity, stp_mode)

    def sell(self, client_id: int, quantity: int = 1, stp_mode: str | None = None) -> Report:
        return self.market("ask", client_id, quantity, stp_mode)

    def place_bid(self, client_id: int, bid: int, quantity: int, stp_mode: str | None = None) -> int:
        return self.limit("bid", client_id, bid, quantity, stp_mode)["order_id"]

    def place_ask(self, client_id: int, ask: int, quantity: int, stp_mode: str | None = None) -> int:
        return self.limit("ask", client_id, ask, quantity, stp_mode)["order_id"]

    def _owned(self, order_id, client_id) -> _Order:
        order = self._orders.get(self._integer(order_id, "order_id"))
        if order is None:
            raise KeyError(f"Order {order_id} is not active.")
        if client_id is not None and order.client_id != self._integer(client_id, "client_id"):
            raise PermissionError(f"Client {client_id} does not own order {order_id}.")
        return order

    def _lifecycle_report(self, order: _Order, operation: str, cancelled: int) -> Report:
        report: Report = {
            "order_id": order.order_id, "client_id": order.client_id, "side": order.side,
            "order_type": operation, "price": order.price,
            "requested_quantity": order.quantity + cancelled, "filled_quantity": 0,
            "resting_quantity": order.quantity, "cancelled_quantity": cancelled,
            "avg_fill_price": None, "trades": [], "stp_cancelled_orders": [],
            "evicted_orders": [], "replaced_order_id": None,
        }
        self._last_report = report
        return report

    def cancel_order(self, order_id: int, client_id: int | None = None) -> Report:
        order = self._owned(order_id, client_id)
        self._remove(order)
        cancelled, order.quantity = order.quantity, 0
        self._refresh(order.side)
        return self._lifecycle_report(order, "cancel", cancelled)

    def modify_order(self, order_id: int, new_quantity: int | None = None,
                     new_price: int | None = None, client_id: int | None = None) -> int | None:
        """Reduce in place (keeps priority); otherwise replace with a new order id."""
        order = self._owned(order_id, client_id)
        quantity = order.quantity if new_quantity is None else self._integer(new_quantity, "new_quantity")
        price = order.price if new_price is None else self._validate_price(order.side, new_price)
        if quantity == 0:
            self.cancel_order(order_id, client_id)
            return None
        if price == order.price and quantity <= order.quantity:
            self._volumes[order.side][price] -= order.quantity - quantity
            cancelled, order.quantity = order.quantity - quantity, quantity
            self._lifecycle_report(order, "modify", cancelled)
            return order_id
        return self._submit(order.side, order.client_id, quantity, price, None, replacement=order)["order_id"]

    # ----- observation ----------------------------------------------------- #

    def get_snapshot(self, depth: int | None = None) -> dict:
        depth = self.num_levels if depth is None else self._integer(depth, "depth")
        return {
            "best_bid": self.best_bid, "best_ask": self.best_ask,
            "mid_price": self.mid_price, "spread": self.spread,
            "bid_window": self.bid_prices, "ask_window": self.ask_prices,
            "bids": [(p, v, len(self._queues["bid"][p])) for p, v in self.levels("bid")[:depth]],
            "asks": [(p, v, len(self._queues["ask"][p])) for p, v in self.levels("ask")[:depth]],
            "total_bid_volume": sum(self._volumes["bid"].values()),
            "total_ask_volume": sum(self._volumes["ask"].values()),
        }

    def get_book_arrays(self, depth: int | None = None) -> dict[str, np.ndarray]:
        """Contiguous tick grids from each anchor outward; invalid prices are masked."""
        depth = self.num_levels if depth is None else self._integer(depth, "depth")
        arrays = {}
        for side, step in (("bid", -1), ("ask", 1)):
            anchor, volumes = self.anchor(side), self._volumes[side]
            prices = np.array([anchor + step * i for i in range(depth)], dtype=object)
            valid = np.array([1 <= p <= self._MAX_INT for p in prices])
            arrays[f"{side}_prices"] = np.where(valid, prices, 0).astype(np.int64)
            arrays[f"{side}_volumes"] = np.array(
                [volumes.get(int(p), 0) if ok else 0 for p, ok in zip(prices, valid)], dtype=np.int64)
            arrays[f"{side}_valid"] = valid
        return arrays
