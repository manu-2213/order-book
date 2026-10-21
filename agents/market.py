"""Accounting, order routing and the shared per-step public state.

The book knows nothing about money. ``Market`` owns every account, keeps the
cash reserved by resting bids and the shares reserved by resting asks as
running totals (updated on place, fill, cancel, self-trade cancel and window
eviction), and settles both sides of every fill. Nobody can short or borrow,
so cash and shares are conserved across the population.

``PublicState`` is refreshed once per step from the book and the step's
trades. Agents read it; none of them recompute averages or flow statistics.
"""

import math

from order_book import OrderBook, Report, Side


class Account:
    __slots__ = ("cash", "shares", "reserved_cash", "reserved_shares")

    def __init__(self, cash: int, shares: int):
        self.cash, self.shares = cash, shares
        self.reserved_cash = self.reserved_shares = 0

    @property
    def free_cash(self) -> int:
        return self.cash - self.reserved_cash

    @property
    def free_shares(self) -> int:
        return self.shares - self.reserved_shares


class PublicState:
    """What everyone can see, computed once per step.

    EWMA statistics at a fast and a slow horizon (mid, squared mid change,
    traded volume), momentum = fast EMA - slow EMA for a fixed set of lookback
    pairs, rolling signed aggressive flow over ``flow_window`` steps with an
    EWMA of its imbalance for the detector, and an attention multiplier that
    rises with relative volatility and volume (activity clustering).
    """

    def __init__(self, mid: int, lookbacks=((5, 30), (10, 60), (20, 120)), fast: int = 10,
                 slow: int = 100, flow_window: int = 20, flow_memory: int = 10,
                 flow_threshold: float = 0.3):
        self.mid = float(mid)
        self.last: int | None = None
        self.best_bid = self.best_ask = None
        self.spread = 0
        self.ma_fast = self.ma_slow = float(mid)
        self.var_fast = self.var_slow = 1.0  # ticks^2 per step
        self.volume_fast = self.volume_slow = 0.0
        self.lookbacks = lookbacks
        self.momentum = [0.0] * len(lookbacks)
        self.imbalance = self.flow_memory = self.flow_signal = 0.0
        self.attention = 1.0
        self.step_volume = 0
        self._a_fast, self._a_slow = 2 / (fast + 1), 2 / (slow + 1)
        self._alphas = [(2 / (f + 1), 2 / (s + 1)) for f, s in lookbacks]
        self._emas = [[float(mid), float(mid)] for _ in lookbacks]
        self._a_memory, self._threshold = 2 / (flow_memory + 1), flow_threshold
        self._buy = [0] * flow_window
        self._sell = [0] * flow_window
        self.buy_flow = self.sell_flow = 0
        self._n = 0

    def update(self, book: OrderBook, buy_volume: int, sell_volume: int, last: int | None) -> None:
        bid, ask = book.best_bid, book.best_ask
        self.best_bid, self.best_ask = bid, ask
        if bid is not None and ask is not None:
            mid, self.spread = (bid + ask) / 2, ask - bid
        else:
            mid = float(last) if last is not None else self.mid
        change, self.mid, self.last = mid - self.mid, mid, last
        a, b = self._a_fast, self._a_slow
        self.ma_fast += a * (mid - self.ma_fast)
        self.ma_slow += b * (mid - self.ma_slow)
        self.var_fast += a * (change * change - self.var_fast)
        self.var_slow += b * (change * change - self.var_slow)
        volume = self.step_volume = buy_volume + sell_volume
        self.volume_fast += a * (volume - self.volume_fast)
        self.volume_slow += b * (volume - self.volume_slow)
        for k, (af, asl) in enumerate(self._alphas):
            ema = self._emas[k]
            ema[0] += af * (mid - ema[0])
            ema[1] += asl * (mid - ema[1])
            self.momentum[k] = ema[0] - ema[1]
        i = self._n % len(self._buy)
        self.buy_flow += buy_volume - self._buy[i]
        self.sell_flow += sell_volume - self._sell[i]
        self._buy[i], self._sell[i] = buy_volume, sell_volume
        total = self.buy_flow + self.sell_flow
        self.imbalance = (self.buy_flow - self.sell_flow) / total if total else 0.0
        self.flow_memory += self._a_memory * (self.imbalance - self.flow_memory)
        strength = (abs(self.flow_memory) - self._threshold) / (1 - self._threshold)
        self.flow_signal = math.copysign(min(strength, 1.0), self.flow_memory) if strength > 0 else 0.0
        self._n += 1
        if self._n * self._a_slow > 1:  # past the slow warm-up; before it the ratios are meaningless
            relative_vol = math.sqrt(self.var_fast / self.var_slow) if self.var_slow > 0 else 1.0
            relative_volume = self.volume_fast / self.volume_slow if self.volume_slow > 0 else 1.0
            self.attention = min(3.0, max(0.5, 1.0 + 0.5 * (relative_vol - 1) + 0.5 * (relative_volume - 1)))


class Market:
    """Routes agent orders to the book and settles cash/shares for both parties.

    In analysis mode every fill is appended to ``tape`` as
    (step, price, quantity, aggressor_side, buyer_id, seller_id); in training
    mode only the rolling aggregates in ``public`` are kept.
    """

    def __init__(self, book: OrderBook, analysis: bool = True, **public_kwargs):
        self.book = book
        self.analysis = analysis
        book.keep_trades = analysis
        self.accounts: dict[int, Account] = {}
        self.public = PublicState(book.initial_mid_price, **public_kwargs)
        self.step = 0
        self.last_price: int | None = None
        self.tape: list[tuple] | None = [] if analysis else None
        self.budget_rejects = {"bid": 0, "ask": 0}
        self.window_rejects = {"bid": 0, "ask": 0}
        self._buy_volume = self._sell_volume = 0

    def open_account(self, cid: int, cash: int, shares: int) -> Account:
        if cid in self.accounts:
            raise ValueError(f"client {cid} already has an account")
        account = self.accounts[cid] = Account(cash, shares)
        return account

    def wealth(self, cid: int) -> float:
        account = self.accounts[cid]
        return account.cash + account.shares * self.public.mid

    def orders(self, cid: int):
        return self.book.get_client_orders(cid)

    # ----- settlement ------------------------------------------------------ #

    def _release(self, order) -> None:
        account = self.accounts[order["client_id"]]
        if order["side"] == "bid":
            account.reserved_cash -= order["price"] * order["quantity"]
        else:
            account.reserved_shares -= order["quantity"]

    def _settle(self, report: Report) -> Report:
        side = report["side"]
        trades = report["trades"]
        if trades:
            accounts = self.accounts
            for t in trades:
                price, quantity = t["price"], t["quantity"]
                notional = price * quantity
                buyer, seller = accounts[t["buyer_id"]], accounts[t["seller_id"]]
                buyer.cash -= notional
                buyer.shares += quantity
                seller.cash += notional
                seller.shares -= quantity
                if side == "bid":  # the maker was a resting ask
                    seller.reserved_shares -= quantity
                else:  # the maker was a resting bid, filled at its own price
                    buyer.reserved_cash -= notional
            self.last_price = price
            if side == "bid":
                self._buy_volume += report["filled_quantity"]
            else:
                self._sell_volume += report["filled_quantity"]
            if self.tape is not None:
                aggressor = "buy" if side == "bid" else "sell"
                self.tape.extend((self.step, t["price"], t["quantity"], aggressor,
                                  t["buyer_id"], t["seller_id"]) for t in trades)
        resting = report["resting_quantity"]
        if resting:
            account = self.accounts[report["client_id"]]
            if side == "bid":
                account.reserved_cash += report["price"] * resting
            else:
                account.reserved_shares += resting
        for order in report["evicted_orders"]:
            self._release(order)
        for order in report["stp_cancelled_orders"]:
            self._release(order)
        return report

    # ----- actions --------------------------------------------------------- #

    def limit(self, cid: int, side: Side, price: int, quantity: int) -> Report | None:
        """Funded limit order; outside-window orders are dropped, never moved."""
        if price < 1 or quantity < 1:
            return None
        edge = self.book.edge(side)
        if price < edge if side == "bid" else price > edge:
            self.window_rejects[side] += 1
            return None
        account = self.accounts[cid]
        cap = account.free_cash // price if side == "bid" else account.free_shares
        if quantity > cap:
            quantity = cap
        if quantity < 1:
            self.budget_rejects[side] += 1
            return None
        return self._settle(self.book.limit(side, cid, price, quantity))

    def ioc(self, cid: int, side: Side, price: int, quantity: int) -> Report | None:
        """Immediate-or-cancel: trade up to ``price`` now, cancel whatever would rest."""
        report = self.limit(cid, side, price, quantity)
        if report is not None and report["resting_quantity"]:
            self.cancel(cid, report["order_id"])
        return report

    def market(self, cid: int, side: Side, quantity: int) -> Report | None:
        account = self.accounts[cid]
        if side == "bid":  # walk the ask levels so the buy is always funded
            cash, affordable = account.free_cash, 0
            for price, volume in self.book.levels("ask"):
                take = min(volume, cash // price, quantity - affordable)
                affordable += take
                cash -= take * price
                if take < volume:
                    break
            quantity = affordable
        else:
            quantity = min(quantity, account.free_shares)
        if quantity < 1:
            self.budget_rejects[side] += 1
            return None
        return self._settle(self.book.market(side, cid, quantity))

    def cancel(self, cid: int, order_id: int) -> None:
        report = self.book.cancel_order(order_id, cid)
        account = self.accounts[cid]
        if report["side"] == "bid":
            account.reserved_cash -= report["price"] * report["cancelled_quantity"]
        else:
            account.reserved_shares -= report["cancelled_quantity"]

    def cancel_all(self, cid: int) -> None:
        for order_id in list(self.book.client_order_ids(cid)):
            self.cancel(cid, order_id)

    def end_step(self) -> None:
        self.public.update(self.book, self._buy_volume, self._sell_volume, self.last_price)
        self._buy_volume = self._sell_volume = 0
        self.step += 1
