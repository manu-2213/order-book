"""The ghost arm: an executor fills against a live copy of Arm 0's book, unseen by everyone else.

``GhostMarket`` forks a ``Market`` and is then driven by it. The background
agents keep acting on the source market, which never learns the executor
exists; every order and cancel that reaches the source book is replayed on the
copy, where the executor also trades. What the executor pays there is pure book
consumption plus natural refill, the Bouchaud-Farmer-Lillo "virtual impact".

The two books diverge as soon as the executor trades, so replaying an order
needs a rule. The rule treats each lot as what it was in Arm 0:

    R1  A lot that traded in Arm 0 takes liquidity. It is replayed immediately-
        or-cancel at its own limit (market orders at no limit) and walks the
        copy's book by the ordinary price-time matching. If the copy has nothing
        at that price the lot lapses: its sender wanted to trade now, it would
        not have rested. This is what keeps a marketable order from turning into
        a resting quote that no owner would ever cancel.
    R2  A lot that rested in Arm 0 makes liquidity. It rests at its own price,
        post-only clamped one tick behind the copy's opposite best if it would
        cross there (the rule the market makers already apply to themselves), and
        dropped if that price is outside the copy's window.
    R3  A cancel, eviction or STP removal in Arm 0 removes the order's counterpart
        in the copy, if it is still there.
    R4  An order that fully traded in Arm 0 but is still resting in the copy has no
        owner who will ever cancel it. These orphans arise when the executor's
        passive order takes the fill the Arm-0 order got. They stay, because the
        executor really did add that much liquidity, but expire after ``orphan_ttl``
        steps, the order of magnitude of a background order's life, so they cannot
        pile up into a wall no real population would maintain.
    R5  Only the executor is budget-checked. The background's accounts in the copy
        are shadows: orders were funded in Arm 0, so a shadow may go negative. Cash
        and shares are still conserved across the copy.

With no executor the copy reproduces the source exactly, lot for lot and tick
for tick. ``stats`` counts every time the rule had to intervene.
"""

import copy
from collections import deque
from dataclasses import dataclass

from order_book.market import Market


@dataclass(frozen=True)
class GhostConfig:
    orphan_ttl: int | None = 50   # steps an orphaned order survives in the copy; None keeps it until it trades


class GhostMarket(Market):
    """A forked copy of ``source``'s book, accounts and public state, replaying its order flow."""

    def __init__(self, source: Market, cfg: GhostConfig | None = None):
        book = copy.deepcopy(source.book)
        super().__init__(book, analysis=False)
        book._trades = []
        self.public = copy.deepcopy(source.public)
        self.accounts = copy.deepcopy(source.accounts)
        self.step, self.last_price = source.step, source.last_price
        self.source = source
        self.cfg = cfg or GhostConfig()
        self.quote_shade = None                     # optional (cid, side, price, qty) -> (price, qty) | None
        self.stats = {"lapsed": 0, "clamped": 0, "dropped": 0, "shaded_away": 0, "orphaned": 0, "expired": 0}
        self._ids = {order_id: order_id for order_id in book._orders}  # source order id -> copy order id
        self._orphans: deque = deque()              # (expiry step, copy order id)
        source.mirror = self

    # ----- replay ----------------------------------------------------------- #

    def on_order(self, report) -> None:
        side, cid, price = report["side"], report["client_id"], report["price"]
        if report["filled_quantity"]:
            self._take(side, cid, price, report["filled_quantity"])
        if report["resting_quantity"]:
            self._make(report["order_id"], side, cid, price, report["resting_quantity"])
        for order in report["evicted_orders"]:
            self._remove_counterpart(order["order_id"])
        for order in report["stp_cancelled_orders"]:
            self._remove_counterpart(order["order_id"])
        if report["trades"]:
            self._mark_orphans(report)

    def on_cancel(self, cid: int, order_id: int) -> None:
        self._remove_counterpart(order_id)

    def _take(self, side: str, cid: int, price: int | None, quantity: int) -> None:
        book = self.book
        report = book.market(side, cid, quantity) if price is None else book.ioc(side, cid, price, quantity)
        self.stats["lapsed"] += report["cancelled_quantity"]
        self._settle(report)

    def _make(self, source_id: int, side: str, cid: int, price: int, quantity: int) -> None:
        book = self.book
        if self.quote_shade is not None:
            shaded = self.quote_shade(cid, side, price, quantity)
            if shaded is None:
                self.stats["shaded_away"] += 1
                return
            price, quantity = shaded
        opposite = book.best_ask if side == "bid" else book.best_bid
        if opposite is not None:
            if side == "bid" and price >= opposite:
                price = opposite - 1
                self.stats["clamped"] += 1
            elif side == "ask" and price <= opposite:
                price = opposite + 1
                self.stats["clamped"] += 1
        edge = book.edge(side)
        if price < 1 or (price < edge if side == "bid" else price > edge):
            self.stats["dropped"] += 1
            return
        self._ids[source_id] = self._settle(book.limit(side, cid, price, quantity))["order_id"]

    def _remove_counterpart(self, source_id: int) -> None:
        copy_id = self._ids.pop(source_id, None)
        if copy_id is not None and self.book.is_active(copy_id):
            self._cancel_copy(copy_id)

    def _cancel_copy(self, copy_id: int) -> None:
        Market.cancel(self, self.book.get_order(copy_id)["client_id"], copy_id)

    def _mark_orphans(self, report) -> None:
        maker_key = "sell_order_id" if report["side"] == "bid" else "buy_order_id"
        source_book, ttl = self.source.book, self.cfg.orphan_ttl
        for trade in report["trades"]:
            source_id = trade[maker_key]
            if not source_book.is_active(source_id):
                copy_id = self._ids.pop(source_id, None)
                if copy_id is not None and self.book.is_active(copy_id):
                    self.stats["orphaned"] += 1
                    if ttl is not None:
                        self._orphans.append((self.step + ttl, copy_id))

    def end_step(self) -> None:
        orphans = self._orphans
        while orphans and orphans[0][0] <= self.step:
            _, copy_id = orphans.popleft()
            if self.book.is_active(copy_id):
                self.stats["expired"] += 1
                self._cancel_copy(copy_id)
        super().end_step()
