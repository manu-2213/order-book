"""Accounting and order routing.

The book knows nothing about money. ``Market`` owns every account, keeps the
cash reserved by resting bids and the shares reserved by resting asks as
running totals (updated on place, fill, cancel, self-trade cancel and window
eviction), and settles both sides of every fill. Nobody can short or borrow,
so cash and shares are conserved across the population.

Trade reports carrying client identities stay inside settlement; agents only
see ``public`` (a ``PublicState`` refreshed once per step) and their own
accounts and orders.
"""

from order_book.book import OrderBook, Report, Side
from order_book.config import PublicStateConfig
from order_book.public_state import PublicState


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


class Market:
    """Routes agent orders to the book and settles cash/shares for both parties.

    In analysis mode every fill is appended to ``tape`` as
    (step, price, quantity, aggressor_side, buyer_id, seller_id); in training
    mode only the rolling aggregates in ``public`` are kept.
    """

    def __init__(self, book: OrderBook, analysis: bool = True, public: PublicStateConfig | None = None):
        self.book = book
        self.analysis = analysis
        book.keep_trades = analysis
        self.accounts: dict[int, Account] = {}
        self.public = PublicState(book.initial_mid_price, public)
        self.step = 0
        self.last_price: int | None = None
        self.tape: list[tuple] | None = [] if analysis else None
        self.budget_rejects = {"bid": 0, "ask": 0}
        self.window_rejects = {"bid": 0, "ask": 0}
        self._buy_volume = self._sell_volume = 0
        self._flow_buy = self._flow_sell = 0       # aggressive volume visible to detectors this step
        self._step_trades: list[tuple] = []        # anonymous (side, price, quantity) aggressive fills
        self.hidden_aggressors: set[int] = set()   # analysis-only counterfactual, see ``hide``

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
            filled = report["filled_quantity"]
            if side == "bid":
                self._buy_volume += filled
            else:
                self._sell_volume += filled
            if report["client_id"] not in self.hidden_aggressors:
                if side == "bid":
                    self._flow_buy += filled
                else:
                    self._flow_sell += filled
                self._step_trades.extend((side, t["price"], t["quantity"]) for t in trades)
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

    def hide(self, cid: int) -> None:
        """Make ``cid``'s aggressive trades invisible to the public flow statistics and detectors.

        The trades still happen, settle, move the price and count in the
        public volume totals; only the signed flow window and the anonymous
        per-step trade list omit them. This deliberately breaks the
        no-identity rule and exists only for the analysis-mode counterfactual
        that separates impact cost from leakage cost.
        """
        self.hidden_aggressors.add(cid)

    def end_step(self) -> None:
        self.public.update(self.book, self._buy_volume, self._sell_volume, self.last_price,
                           self._flow_buy, self._flow_sell, self._step_trades)
        self._buy_volume = self._sell_volume = self._flow_buy = self._flow_sell = 0
        self._step_trades = []
        self.step += 1
